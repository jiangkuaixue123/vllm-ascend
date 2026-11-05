/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <initializer_list>
#include "register/op_def_registry.h"

namespace ops {
class HcPreDelayed : public OpDef {
public:
    explicit HcPreDelayed(const char *name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        for (const char *input : {"hc_fn", "hc_scale", "hc_base", "previous_pre"}) {
            this->Input(input).ParamType(REQUIRED).DataType({ge::DT_FLOAT})
                .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        }
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        for (const char *output : {"next_pre", "post", "comb_frag", "projection", "rstd"}) {
            this->Output(output).ParamType(REQUIRED).DataType({ge::DT_FLOAT})
                .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        }
        // Attribute order intentionally agrees with the shared legacy tiler.
        this->Attr("hc_mult").AttrType(OPTIONAL).Int(4);
        this->Attr("hc_sinkhorn_iters").AttrType(OPTIONAL).Int(20);
        this->Attr("hc_eps").AttrType(OPTIONAL).Float(1e-6f);
        this->Attr("norm_eps").AttrType(OPTIONAL).Float(1e-20f);
        this->AICore().AddConfig("ascend910_93");
    }
};
OP_ADD(HcPreDelayed);
}  // namespace ops
