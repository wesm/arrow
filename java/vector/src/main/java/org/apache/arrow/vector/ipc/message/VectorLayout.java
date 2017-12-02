/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.arrow.vector.ipc.message;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.google.common.base.Preconditions;
import com.google.flatbuffers.FlatBufferBuilder;

public class VectorLayout {

  private static final VectorLayout VALIDITY_VECTOR = new VectorLayout(ArrowVectorType.VALIDITY, 1);
  private static final VectorLayout OFFSET_VECTOR = new VectorLayout(ArrowVectorType.OFFSET, 32);
  private static final VectorLayout TYPE_VECTOR = new VectorLayout(ArrowVectorType.TYPE, 32);
  private static final VectorLayout BOOLEAN_VECTOR = new VectorLayout(ArrowVectorType.DATA, 1);
  private static final VectorLayout VALUES_128 = new VectorLayout(ArrowVectorType.DATA, 128);
  private static final VectorLayout VALUES_64 = new VectorLayout(ArrowVectorType.DATA, 64);
  private static final VectorLayout VALUES_32 = new VectorLayout(ArrowVectorType.DATA, 32);
  private static final VectorLayout VALUES_16 = new VectorLayout(ArrowVectorType.DATA, 16);
  private static final VectorLayout VALUES_8 = new VectorLayout(ArrowVectorType.DATA, 8);

  public static VectorLayout typeVector() {
    return TYPE_VECTOR;
  }

  public static VectorLayout offsetVector() {
    return OFFSET_VECTOR;
  }

  public static VectorLayout dataVector(int typeBitWidth) {
    switch (typeBitWidth) {
      case 8:
        return VALUES_8;
      case 16:
        return VALUES_16;
      case 32:
        return VALUES_32;
      case 64:
        return VALUES_64;
      case 128:
        return VALUES_128;
      default:
        throw new IllegalArgumentException("only 8, 16, 32, or 64 bits supported");
    }
  }

  public static VectorLayout booleanVector() {
    return BOOLEAN_VECTOR;
  }

  public static VectorLayout validityVector() {
    return VALIDITY_VECTOR;
  }

  public static VectorLayout byteVector() {
    return dataVector(8);
  }

  private final short typeBitWidth;

  private final ArrowVectorType type;

  @JsonCreator
  private VectorLayout(@JsonProperty("type") ArrowVectorType type, @JsonProperty("typeBitWidth") int typeBitWidth) {
    super();
    this.type = Preconditions.checkNotNull(type);
    this.typeBitWidth = (short) typeBitWidth;
    if (typeBitWidth <= 0) {
      throw new IllegalArgumentException("bitWidth invalid: " + typeBitWidth);
    }
  }

  public int getTypeBitWidth() {
    return typeBitWidth;
  }

  public ArrowVectorType getType() {
    return type;
  }

  @Override
  public String toString() {
    return String.format("%s(%s)", type, typeBitWidth);
  }

  @Override
  public int hashCode() {
    return 31 * (31 + type.hashCode()) + typeBitWidth;
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (obj == null) {
      return false;
    }
    if (getClass() != obj.getClass()) {
      return false;
    }
    VectorLayout other = (VectorLayout) obj;
    return type.equals(other.type) && (typeBitWidth == other.typeBitWidth);
  }
}
