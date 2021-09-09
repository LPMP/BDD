#include "bdd_cuda_base.h"

namespace LPMP {

    bdd_cuda_base::bdd_cuda_base(bdd_collection& bdd_col)
    {
        for(size_t bdd_idx=0; bdd_idx<bdd_col.nr_bdds(); ++bdd_idx)
        {
            assert(bdd_col.is_qbdd(bdd_idx));
            for(size_t bdd_node_idx=0; bdd_node_idx<bdd_col.nr_bdd_nodes(bdd_idx); ++bdd_node_idx)
            {
                const auto cur_instr bdd_col(bdd_idx, bdd_node_idx);
                if(!cur_instr.is_terminal())
                {
                    const size_t var = cur_instr.index;
                    const auto lo_instr = bdd_col(bdd_idx, cur_instr.lo);
                    const auto hi_instr = bdd_col(bdd_idx, cur_instr.hi);
                }
            }
        }
    }
}
