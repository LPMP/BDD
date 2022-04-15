#include "specialized_solvers/mrf_input.h"

namespace LPMP {

    size_t mrf_input::nr_variables() const
    {
        return unaries_.size();
    }

    size_t mrf_input::nr_labels(const size_t var) const
    {
        assert(var < nr_variables());
        return unaries_.size(var);
    }

    double mrf_input::unary(const size_t var, const size_t label) const
    {
        assert(var < nr_variables());
        assert(label < nr_labels(var));
        return unaries_(var,label);
    }

    double& mrf_input::unary(const size_t var, const size_t label)
    {
        assert(var < nr_variables());
        assert(label < nr_labels(var));
        return unaries_(var,label);
    }

    size_t mrf_input::nr_pairwise_potentials() const
    {
        assert(pairwise_.size() == pairwise_variables_.size());
        return pairwise_.size();
    }

    double mrf_input::pairwise(const size_t pairwise_pot, const std::array<size_t,2> labels) const
    {
        assert(pairwise_pot < nr_pairwise_potentials());
        assert(labels[0] < nr_labels(pairwise_variables(pairwise_pot)[0]));
        assert(labels[1] < nr_labels(pairwise_variables(pairwise_pot)[1]));

        const size_t lidx = labels[0] + labels[1] * nr_labels(pairwise_variables(pairwise_pot)[0]);

        return pairwise_(pairwise_pot, lidx);
    }

    double& mrf_input::pairwise(const size_t pairwise_pot, const std::array<size_t,2> labels)
    {
        assert(pairwise_pot < nr_pairwise_potentials());
        assert(labels[0] < nr_labels(pairwise_variables(pairwise_pot)[0]));
        assert(labels[1] < nr_labels(pairwise_variables(pairwise_pot)[1]));

        const size_t lidx = labels[0] + labels[1] * nr_labels(pairwise_variables(pairwise_pot)[0]);

        return pairwise_(pairwise_pot, lidx);
    }

    std::array<size_t,2> mrf_input::pairwise_variables(const size_t pairwise_pot) const
    {
        assert(pairwise_pot < nr_pairwise_potentials());
        return pairwise_variables_[pairwise_pot];
    }

    ILP_input mrf_input::convert_to_ilp() const
    {
        ILP_input ilp;
        // first construct unary variables
        std::vector<size_t> unary_offsets;
        unary_offsets.reserve(nr_variables());
        for(size_t i=0; i<nr_variables(); ++i)
        {
            unary_offsets.push_back(ilp.nr_variables());
            for(size_t l=0; l<nr_labels(i); ++l)
            {
                const size_t var = ilp.add_new_variable("x_" + std::to_string(i) + "_" + std::to_string(l));
                ilp.add_to_objective(unary(i,l), var);
            }
        }

        // construct pairwise variables
        std::vector<size_t> pairwise_offsets;
        pairwise_offsets.reserve(nr_pairwise_potentials());
        for(size_t p=0; p<nr_pairwise_potentials(); ++p)
        {
            pairwise_offsets.push_back(ilp.nr_variables());
            const auto [i,j] = pairwise_variables(p);
            // TODO: add better handling for potts potentials
            for(size_t l_i=0; l_i<nr_labels(i); ++l_i)
                for(size_t l_j=0; l_j<nr_labels(j); ++l_j)
                {
                    const size_t var = ilp.add_new_variable("x_" + std::to_string(i) + "_" + std::to_string(j) + "_" + std::to_string(l_i) + "_" + std::to_string(l_j));
                    ilp.add_to_objective(pairwise(p, {l_i,l_j}), var);
                }
        }

        std::vector<int> coeffs;
        std::vector<size_t> vars;

        // construct simplex constraints for unary variables
        for(size_t i=0; i<nr_variables(); ++i)
        {
            coeffs.clear();
            vars.clear();
            for(size_t l=0; l<nr_labels(i); ++l)
            {
                coeffs.push_back(1);
                vars.push_back(unary_offsets[i] + l);
            }

            ilp.add_constraint(coeffs, vars, ILP_input::inequality_type::equal, 1);
        }

        // construct simplex constraints for pairwise variabels
        for(size_t p=0; p<nr_pairwise_potentials(); ++p)
        {
            vars.clear();
            coeffs.clear();
            const auto [i,j] = pairwise_variables(p);
            for(size_t l_i=0; l_i<nr_labels(i); ++l_i)
                for(size_t l_j=0; l_j<nr_labels(j); ++l_j)
                {
                    coeffs.push_back(1);
                    vars.push_back(pairwise_offsets[p] + l_i*nr_labels(j) + l_j);
                }

            ilp.add_constraint(coeffs, vars, ILP_input::inequality_type::equal, 1);
        }

        // construct marginalization constraints
        for(size_t p=0; p<nr_pairwise_potentials(); ++p)
        {
            const auto [i,j] = pairwise_variables(p);
            for(size_t l_i=0; l_i<nr_labels(i); ++l_i)
            {
                vars.clear();
                coeffs.clear();
                vars.push_back(unary_offsets[i]+l_i);
                coeffs.push_back(-1);
                for(size_t l_j=0; l_j<nr_labels(j); ++l_j)
                {
                    vars.push_back(pairwise_offsets[p] + l_i*nr_labels(j) + l_j);
                    coeffs.push_back(1);
                }
                ilp.add_constraint(coeffs, vars, ILP_input::inequality_type::equal, 0);
            }

            for(size_t l_j=0; l_j<nr_labels(j); ++l_j)
            {
                vars.clear();
                coeffs.clear();
                vars.push_back(unary_offsets[j]+l_j);
                coeffs.push_back(-1);
                for(size_t l_i=0; l_i<nr_labels(i); ++l_i)
                {
                    vars.push_back(pairwise_offsets[p] + l_i*nr_labels(j) + l_j);
                    coeffs.push_back(1);
                }
                ilp.add_constraint(coeffs, vars, ILP_input::inequality_type::equal, 0);
            }
        }

        return ilp;
    }
}
