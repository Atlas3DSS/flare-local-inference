"""Remove descriptive ELF names while preserving CUDA section and symbol indices."""

import struct


def redact_symbols(image):
    """Return a cubin and its original-to-exported function/global lookup names.

    Only nonallocated string tables and their references change. Code, relocations,
    CUDA metadata, symbol order, and section order retain their captured layout.
    """
    if len(image) < 64 or image[:6] != b"\x7fELF\x02\x01":
        raise ValueError("expected an ELF64 little-endian cubin")
    offset = struct.unpack_from("<Q", image, 40)[0]
    entry_size, count, names = struct.unpack_from("<HHH", image, 58)
    if entry_size != 64 or not 0 < names < count or offset + count * entry_size > len(image):
        raise ValueError("invalid cubin ELF section table")
    sections = [struct.unpack_from("<IIQQQQIIQQ", image, offset + i * entry_size) for i in range(count)]

    string_data = {}

    def contents(index):
        if index in string_data:
            return string_data[index]
        section = sections[index]
        start, size = section[4:6]
        if section[1] == 8 or start + size > len(image):
            raise ValueError("invalid cubin section extent")
        string_data[index] = image[start:start + size]
        return string_data[index]

    def string(table, index):
        data = contents(table)
        if sections[table][1] != 3 or not 0 <= index < len(data):
            raise ValueError("invalid cubin string reference")
        end = data.find(b"\0", index)
        if end < 0:
            raise ValueError("unterminated cubin string")
        return data[index:end]

    symbols = []
    for table in sections:
        # CUDA's alternate-code symbol table uses the ELF64 symbol layout.
        if table[1] not in (2, 0x70000085):
            continue
        if not 0 < table[6] < count or table[9] != 24 or table[5] % 24:
            raise ValueError("invalid cubin symbol table")
        if table[4] + table[5] > len(image):
            raise ValueError("invalid cubin symbol table extent")
        for position in range(table[4], table[4] + table[5], 24):
            name, info, _, section, _, _ = struct.unpack_from("<IBBHQQ", image, position)
            symbols.append((position, table[6], string(table[6], name), info & 15, section))

    existing = {name for _, _, name, _, _ in symbols}
    aliases = {}
    for _, _, name, kind, section in symbols:
        if (kind not in (1, 2) or not 0 < section < count or not name
                or name.startswith((b".nv.", b"__nv_", b"__cuda")) or name in aliases):
            continue
        index = len(aliases)
        alias = f"_f{index}".encode()
        while alias in existing:
            alias = b"_" + alias
        aliases[name] = alias
        existing.add(alias)

    def renamed(name):
        if name in aliases:
            return aliases[name]
        for index, byte in enumerate(name):
            if byte == ord(".") and name[index + 1:] in aliases:
                return name[:index + 1] + aliases[name[index + 1:]]
        return name

    tables, positions = {}, {}

    def intern(table, value):
        data = tables.setdefault(table, bytearray(b"\0"))
        entries = positions.setdefault(table, {b"": 0})
        if value not in entries:
            entries[value] = len(data)
            data.extend(value + b"\0")
        return entries[value]

    result = bytearray(image)
    for i, section in enumerate(sections):
        name = renamed(string(names, section[0]))
        struct.pack_into("<I", result, offset + i * entry_size, intern(names, name))
    for position, table, name, kind, _ in symbols:
        # STT_FILE is diagnostic metadata, including when its string is suffix-shared.
        struct.pack_into("<I", result, position, intern(table, b"" if kind == 4 else renamed(name)))

    program_offset = struct.unpack_from("<Q", image, 32)[0]
    program_size, program_count = struct.unpack_from("<HH", image, 54)
    if program_count and (program_size != 56 or program_offset + program_count * program_size > len(image)):
        raise ValueError("invalid cubin program table")
    for index, data in tables.items():
        section = sections[index]
        start, size = section[4:6]
        if section[2] & 2:
            raise ValueError("cannot redact a device-allocated cubin string table")
        if start < offset + count * entry_size and offset < start + size:
            raise ValueError("cubin string table overlaps section headers")
        if start < 64 or (start < program_offset + program_count * program_size and program_offset < start + size):
            raise ValueError("cubin string table overlaps ELF headers")
        for i, other in enumerate(sections):
            if i != index and other[1] != 8 and other[5] and start < other[4] + other[5] and other[4] < start + size:
                raise ValueError("cubin string table overlaps another section")
        for i in range(program_count):
            kind, _, source, _, _, length, _, _ = struct.unpack_from("<IIQQQQQQ", image, program_offset + i * program_size)
            if kind == 1 and start < source + length and source < start + size:
                raise ValueError("cannot redact a loadable cubin string table")
        result[start:start + size] = bytes(size)
        # Appending avoids moving code or assuming that suffix-shared names fit in place.
        struct.pack_into("<QQ", result, offset + index * entry_size + 24, len(result), len(data))
        result.extend(data)
    return bytes(result), {name.decode(): alias.decode() for name, alias in aliases.items()}
