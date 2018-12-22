import os.path as op
import sys
import io
import sqlite3
from diskcache import Disk, UNKNOWN
from . import utils

if sys.hexversion < 0x03000000:
    TextType = unicode  # pylint: disable=invalid-name,undefined-variable
    BytesType = str
    INT_TYPES = int, long  # pylint: disable=undefined-variable
    range = xrange  # pylint: disable=redefined-builtin,invalid-name,undefined-variable
    io_open = io.open  # pylint: disable=invalid-name
else:
    TextType = str
    BytesType = bytes
    INT_TYPES = (int,)
    io_open = open # pylint: disable=invalid-name

MODE_NONE = 0
MODE_RAW = 1
MODE_BINARY = 2
MODE_TEXT = 3
MODE_PICKLE = 4

class Lz4Disk(Disk):
    "Cache key and value serialization for SQLite database and files."
    def __init__(self, directory, min_file_size=0, pickle_protocol=0):
        super(Lz4Disk, self).__init__(directory, min_file_size, pickle_protocol)

    def put(self, key):
        """Convert `key` to fields key and raw for Cache table.
        :param key: key to convert
        :return: (database key, raw boolean) pair
        """
        # pylint: disable=bad-continuation,unidiomatic-typecheck
        type_key = type(key)

        if type_key is BytesType:
            return sqlite3.Binary(key), True
        elif ((type_key is TextType)
                or (type_key in INT_TYPES
                    and -9223372036854775808 <= key <= 9223372036854775807)
                or (type_key is float)):
            return key, True
        else:
            data = utils.dumps(key)
            result = pickletools.optimize(data)
            return sqlite3.Binary(result), False


    def get(self, key, raw):
        """Convert fields `key` and `raw` from Cache table to key.
        :param key: database key to convert
        :param bool raw: flag indicating raw database storage
        :return: corresponding Python key
        """
        # pylint: disable=no-self-use,unidiomatic-typecheck
        if raw:
            return BytesType(key) if type(key) is sqlite3.Binary else key
        else:
            return utils.loads(key)


    def store(self, value, read, key=UNKNOWN):
        """Convert `value` to fields size, mode, filename, and value for Cache
        table.
        :param value: value to convert
        :param bool read: True when value is file-like object
        :param key: key for item (default UNKNOWN)
        :return: (size, mode, filename, value) tuple for Cache table
        """
        # pylint: disable=unidiomatic-typecheck
        type_value = type(value)
        min_file_size = self.min_file_size

        if ((type_value is TextType and len(value) < min_file_size)
                or (type_value in INT_TYPES
                    and -9223372036854775808 <= value <= 9223372036854775807)
                or (type_value is float)):
            return 0, MODE_RAW, None, value
        elif type_value is BytesType:
            if len(value) < min_file_size:
                return 0, MODE_RAW, None, sqlite3.Binary(value)
            else:
                filename, full_path = self.filename(key, value)

                with open(full_path, 'wb') as writer:
                    writer.write(value)

                return len(value), MODE_BINARY, filename, None
        elif type_value is TextType:
            filename, full_path = self.filename(key, value)

            with io_open(full_path, 'w', encoding='UTF-8') as writer:
                writer.write(value)

            size = op.getsize(full_path)
            return size, MODE_TEXT, filename, None
        elif read:
            size = 0
            reader = ft.partial(value.read, 2 ** 22)
            filename, full_path = self.filename(key, value)

            with open(full_path, 'wb') as writer:
                for chunk in iter(reader, b''):
                    size += len(chunk)
                    writer.write(chunk)

            return size, MODE_BINARY, filename, None
        else:
            result = utils.dumps(value)

            if len(result) < min_file_size:
                return 0, MODE_PICKLE, None, sqlite3.Binary(result)
            else:
                filename, full_path = self.filename(key, value)

                with open(full_path, 'wb') as writer:
                    writer.write(result)

                return len(result), MODE_PICKLE, filename, None


    def fetch(self, mode, filename, value, read):
        """Convert fields `mode`, `filename`, and `value` from Cache table to
        value.
        :param int mode: value mode raw, binary, text, or pickle
        :param str filename: filename of corresponding value
        :param value: database value
        :param bool read: when True, return an open file handle
        :return: corresponding Python value
        """
        # pylint: disable=no-self-use,unidiomatic-typecheck
        if mode == MODE_RAW:
            return BytesType(value) if type(value) is sqlite3.Binary else value
        elif mode == MODE_BINARY:
            if read:
                return open(op.join(self._directory, filename), 'rb')
            else:
                with open(op.join(self._directory, filename), 'rb') as reader:
                    return reader.read()
        elif mode == MODE_TEXT:
            full_path = op.join(self._directory, filename)
            with io_open(full_path, 'r', encoding='UTF-8') as reader:
                return reader.read()
        elif mode == MODE_PICKLE:
            if value is None:
                with open(op.join(self._directory, filename), 'rb') as reader:
                    return utils.loads(reader.read())
            else:
                return utils.loads(value)
