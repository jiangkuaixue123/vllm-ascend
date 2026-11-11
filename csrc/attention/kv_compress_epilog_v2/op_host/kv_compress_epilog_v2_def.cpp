// SPDX-License-Identifier: CANN-Open-Software-License-Agreement-Version-2.0
#include "register/op_def_registry.h"

namespace ops {
class KvCompressEpilogV2 : public OpDef {
public:
    explicit KvCompressEpilogV2(const char *name) : OpDef(name)
    {
        this->Input("cache").ParamType(REQUIRED).DataType({ge::DT_UINT8, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_BF16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("slot_mapping").ParamType(REQUIRED).DataType({ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("cache").ParamType(REQUIRED).DataType({ge::DT_UINT8, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("quant_mode").AttrType(REQUIRED).Int();
        this->AICore().AddConfig("ascend950");
    }
};
OP_ADD(KvCompressEpilogV2);
}  // namespace ops
