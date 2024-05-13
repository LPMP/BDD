#include "bdd_solver/bdd_solver.h"

namespace LPMP {

    class graph_matching_bdd_solver : public bdd_solver
    {
    public:
        using bdd_solver::bdd_solver;
        virtual ILP_input read_ILP(const nlohmann::json &config) override;
    };
}