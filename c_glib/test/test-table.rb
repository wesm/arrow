# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

class TestTable < Test::Unit::TestCase
  include Helper::Buildable
  include Helper::Omittable

  sub_test_case(".new") do
    def setup
      @fields = [
        Arrow::Field.new("visible", Arrow::BooleanDataType.new),
        Arrow::Field.new("valid", Arrow::BooleanDataType.new),
      ]
      @schema = Arrow::Schema.new(@fields)
    end

    def dump_table(table)
      table.n_columns.times.collect do |i|
        field = table.schema.get_field(i)
        chunked_array = table.get_column_data(i)
        values = []
        chunked_array.chunks.each do |chunk|
          chunk.length.times do |j|
            values << chunk.get_value(j)
          end
        end
        [
          field.name,
          values,
        ]
      end
    end

    def test_arrays
      require_gi_bindings(3, 3, 1)
      arrays = [
        build_boolean_array([true]),
        build_boolean_array([false]),
      ]
      table = Arrow::Table.new(@schema, arrays)
      assert_equal([
                     ["visible", [true]],
                     ["valid", [false]],
                   ],
                   dump_table(table))
    end

    def test_chunked_arrays
      require_gi_bindings(3, 3, 1)
      arrays = [
        Arrow::ChunkedArray.new([build_boolean_array([true]),
                                 build_boolean_array([false])]),
        Arrow::ChunkedArray.new([build_boolean_array([false]),
                                 build_boolean_array([true])]),
      ]
      table = Arrow::Table.new(@schema, arrays)
      assert_equal([
                     ["visible", [true, false]],
                     ["valid", [false, true]],
                   ],
                   dump_table(table))
    end

    def test_record_batches
      require_gi_bindings(3, 3, 1)
      record_batches = [
        build_record_batch({
                             "visible" => build_boolean_array([true]),
                             "valid" => build_boolean_array([false])
                           }),
        build_record_batch({
                             "visible" => build_boolean_array([false]),
                             "valid" => build_boolean_array([true])
                           }),
      ]
      table = Arrow::Table.new(@schema, record_batches)

      assert_equal([
                     ["visible", [true, false]],
                     ["valid", [false, true]],
                   ],
                   dump_table(table))
    end
  end

  sub_test_case("instance methods") do
    def setup
      fields = [
        Arrow::Field.new("visible", Arrow::BooleanDataType.new),
        Arrow::Field.new("valid", Arrow::BooleanDataType.new),
      ]
      schema = Arrow::Schema.new(fields)
      columns = [
        build_boolean_array([true]),
        build_boolean_array([false]),
      ]
      @table = Arrow::Table.new(schema, columns)
    end

    def test_equal
      fields = [
        Arrow::Field.new("visible", Arrow::BooleanDataType.new),
        Arrow::Field.new("valid", Arrow::BooleanDataType.new),
      ]
      schema = Arrow::Schema.new(fields)
      columns = [
        build_boolean_array([true]),
        build_boolean_array([false]),
      ]
      other_table = Arrow::Table.new(schema, columns)
      assert_equal(@table, other_table)
    end

    def test_schema
      assert_equal(["visible", "valid"],
                   @table.schema.fields.collect(&:name))
    end

    def test_column_data
      assert_equal([
                     Arrow::ChunkedArray.new([build_boolean_array([true])]),
                     Arrow::ChunkedArray.new([build_boolean_array([false])]),
                   ],
                   [
                     @table.get_column_data(0),
                     @table.get_column_data(-1),
                   ])
    end

    def test_n_columns
      assert_equal(2, @table.n_columns)
    end

    def test_n_rows
      assert_equal(1, @table.n_rows)
    end

    def test_add_column
      field = Arrow::Field.new("added", Arrow::BooleanDataType.new)
      chunked_array = Arrow::ChunkedArray.new([build_boolean_array([true])])
      new_table = @table.add_column(1, field, chunked_array)
      assert_equal(["visible", "added", "valid"],
                   new_table.schema.fields.collect(&:name))
    end

    def test_remove_column
      new_table = @table.remove_column(0)
      assert_equal(["valid"],
                   new_table.schema.fields.collect(&:name))
    end

    def test_replace_column
      field = Arrow::Field.new("added", Arrow::BooleanDataType.new)
      chunked_array = Arrow::ChunkedArray.new([build_boolean_array([true])])
      new_table = @table.replace_column(0, field, chunked_array)
      assert_equal(["added", "valid"],
                   new_table.schema.fields.collect(&:name))
    end

    def test_to_s
      table = build_table("valid" => build_boolean_array([true, false, true]))
      assert_equal(<<-TABLE, table.to_s)
valid: bool
----
valid:
  [
    [
      true,
      false,
      true
    ]
  ]
      TABLE
    end

    def test_concatenate
      table = build_table("visible" => build_boolean_array([true, false, true, false]))
      table1 = build_table("visible" => build_boolean_array([true]))
      table2 = build_table("visible" => build_boolean_array([false, true]))
      table3 = build_table("visible" => build_boolean_array([false]))
      assert_equal(table, table1.concatenate([table2, table3]))
    end

    sub_test_case("#slice") do
      test("offset: positive") do
        visibles = [true, false, true]
        table = build_table("visible" => build_boolean_array(visibles))
        assert_equal(build_table("visible" => build_boolean_array([false, true])),
                     table.slice(1, 2))
      end

      test("offset: negative") do
        visibles = [true, false, true]
        table = build_table("visible" => build_boolean_array(visibles))
        assert_equal(build_table("visible" => build_boolean_array([false, true])),
                     table.slice(-2, 2))
      end
    end
  end
end
