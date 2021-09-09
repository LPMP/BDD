#pragma once

#include "bdd_cuda_base.h"

namespace LPMP {

    class bdd_cuda_parallel_mma_sorting : public bdd_cuda_base {
        public:
            bdd_cuda_base::bdd_cuda_base; // inherit base constructor

            void iteration();

        private:

    }
}
