import pyarrow as pa

# Write with V6 as default format

# schema1 = pa.schema([pa.field('f0', pa.utf8())])
# schema1.serialize()
# with open('schema_v6.arrow', 'wb') as f:
#     f.write(schema1.serialize().to_pybytes())

# Recompile C++ and write with V5 metadata, but use UnknownType for ints
schema2 = pa.schema([pa.field('f0', pa.int32())])
schema2.serialize()
with open('schema_unknown_type.arrow', 'wb') as f:
    f.write(schema2.serialize().to_pybytes())
