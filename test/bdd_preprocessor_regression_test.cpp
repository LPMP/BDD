#include <vector>
#include <string>
#include "ILP_input.h"
#include "ILP_parser.h"
#include "bdd_preprocessor.h"
#include "test.h"
#include "two_dimensional_variable_array.hxx"

using namespace LPMP;

std::vector<std::string> test_instances = {
    // from hotel
    "energy_hotel_frame15frame99.lp",
    // from house
    "energy_house_frame15frame105.lp",
    // from cell tracking AISTATS
    "drosophila.lp",
    // from shape matching TOSCA
    "000000880800_241600_cat0_200_cat6_200_.lp",
    // from color-seg-n4
    "pfau-small.lp",
    // from worms graph matching Kainmueller et al
    "worm01-16-03-11-1745.lp",
    // from protein-folding
    "1CKK.lp"
    };

int main(int argc, char** argv)
{
    for(const std::string& instance : test_instances)
    {
        std::cout << "checking bdd preprocessor on " << instance << "\n";
        const std::string full_path = BDD_SOLVER_REGRESSION_TEXT_DIR + instance;
        ILP_input ilp = ILP_parser::parse_file(full_path);
        bdd_preprocessor bdd_pre;
        const two_dim_variable_array<size_t> ilp_to_bdd = bdd_pre.add_ilp(ilp);
        const auto& bdd_col = bdd_pre.get_bdd_collection();
        test(ilp_to_bdd.size() == ilp.constraints().size());
        test(bdd_col.nr_bdds() == ilp.constraints().size());
        for(size_t i=0; i<ilp_to_bdd.size(); ++i)
        {
            const auto &constr = ilp.constraints()[i];
            test(ilp_to_bdd.size(i) == 1); // all inequalities can be represented by a single BDD for the regression instances
            const size_t bdd_nr = ilp_to_bdd(i,0);
            test(bdd_nr == i); // for now, only true for no parallelization
            const auto bdd_vars = bdd_col.variables(bdd_nr);
            test(bdd_vars.size() == constr.coefficients.size() && constr.coefficients.size() == constr.monomials.size());
            for(size_t j=0; j<bdd_vars.size(); ++j)
            {
                test(constr.monomials.size(j) == 1);
                //std::cout << bdd_vars[j] << " = " << constr.monomials(j,0) << "\n";
                test(bdd_vars[j] == constr.monomials(j,0));
            }
            //std::cout << "\n";
        }
    }
}
