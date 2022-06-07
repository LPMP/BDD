#include "ILP_input.h"
#include <Eigen/Eigen>
#include "cuthill-mckee.h"
#include "bfs_ordering.hxx"
#include "minimum_degree_ordering.hxx"
#include "time_measure_util.h"
#include "two_dimensional_variable_array.hxx"
#include "union_find.hxx"
#include <iostream>
#include <limits>
#include <unordered_set>

namespace LPMP { 
    bool ILP_input::constraint::monomials_cmp(const two_dim_variable_array<size_t>& monomials, const size_t idx1, const size_t idx2)
    {
        assert(monomials.size(idx1) > 0);
        assert(monomials.size(idx2) > 0);
        const size_t first_var1 = monomials(idx1,0);
        const size_t last_var1 = monomials(idx1,monomials.size(idx1)-1);
        const size_t first_var2 = monomials(idx2,0);
        const size_t last_var2 = monomials(idx2,monomials.size(idx2)-1);

        if(first_var1 != first_var2)
            return first_var1 < first_var2;
        else if(last_var1 != last_var2)
            return last_var1 < last_var2;
        else if(monomials.size(idx1) != monomials.size(idx2))
            return monomials.size(idx1) < monomials.size(idx2);
        else
            for(size_t i=0; i<monomials.size(idx1); ++i)
                if(monomials(idx1,i) != monomials(idx2,i))
                    return monomials(idx1,i) < monomials(idx2,i);
        return idx1 < idx2;
    }

    void ILP_input::constraint::normalize()
    {
        assert(coefficients.size() == monomials.size());
        for(size_t monomial_idx=0; monomial_idx<monomials.size(); ++monomial_idx)
        {
            assert(monomials.size(monomial_idx) > 0);
            std::sort(monomials.begin(monomial_idx), monomials.end(monomial_idx));
        }
        auto sort_pred = [&](const size_t idx1, const size_t idx2)
        {
            return monomials_cmp(monomials, idx1, idx2);
        };
        std::vector<size_t> idx(monomials.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), sort_pred);

        two_dim_variable_array<size_t> sorted_monomials;
        std::vector<int> sorted_coefficients;
        sorted_coefficients.reserve(coefficients.size());
        for(size_t i=0; i<idx.size(); ++i)
        {
            sorted_monomials.push_back(monomials.begin(idx[i]), monomials.end(idx[i]));
            sorted_coefficients.push_back(coefficients[idx[i]]);
        }

        std::swap(coefficients, sorted_coefficients);
        std::swap(monomials, sorted_monomials);
    }

    bool ILP_input::constraint::is_normalized() const
    {
        for(size_t i=0; i+1<monomials.size(); ++i)
            if(!monomials_cmp(monomials, i, i+1))
                return false;
        return true;
    }

    bool ILP_input::constraint::is_linear() const
    {
        for(size_t monomial_idx=0; monomial_idx<monomials.size(); ++monomial_idx)
            if(monomials.size(monomial_idx) != 1)
                return false;
        return true;
    }

    bool ILP_input::constraint::is_simplex() const
    {
        if(ineq != inequality_type::equal)
            return false;
        if(right_hand_side == 0)
            return false;
        if(!is_linear())
            return false;
        for(const int c : coefficients)
            if(c != right_hand_side)
                return false;
        return true;
    }

    bool ILP_input::constraint::distinct_variables() const
    {
        std::unordered_set<size_t> vars;
        for(size_t monomial_idx=0; monomial_idx<monomials.size(); ++monomial_idx)
        {
            for(size_t i=0; i<monomials.size(monomial_idx); ++i)
            {
                const size_t var = monomials(monomial_idx, i);
                if(vars.count(var) > 0)
                    return false;
                vars.insert(var);
            }
        }
        return true;
    }

    void ILP_input::normalize()
    {
#pragma omp parallel for
        for(size_t ineq_nr=0; ineq_nr<constraints_.size(); ++ineq_nr)
            constraints_[ineq_nr].normalize();
    }

    bool ILP_input::is_normalized() const
    {
        for(size_t ineq_nr=0; ineq_nr<constraints_.size(); ++ineq_nr)
            if(!constraints_[ineq_nr].is_normalized())
                return false;
        return true;
    }

    std::vector<size_t> ILP_input::variables(const size_t ineq_nr) const
    {
        assert(ineq_nr < constraints_.size());
        std::vector<size_t> vars;
        for(size_t monomial_idx=0; monomial_idx<constraints_[ineq_nr].monomials.size(); ++monomial_idx)
        {
            for(size_t var_idx=0; var_idx<constraints_[ineq_nr].monomials.size(monomial_idx); ++var_idx)
            {
                const size_t var = constraints_[ineq_nr].monomials(monomial_idx, var_idx);
                vars.push_back(var);
            }
        }
        std::sort(vars.begin(), vars.end());
        vars.erase(std::unique(vars.begin(), vars.end()), vars.end());
        return vars;
    }

    size_t ILP_input::add_new_variable(const std::string& var)
    {
        assert(!var_exists(var));
        const size_t var_index = var_name_to_index_.size();
        var_name_to_index_.insert({var, var_index});
        assert(var_index_to_name_.size() == var_index);
        var_index_to_name_.push_back(var);
        if(objective_.size() <= var_index) // variables with 0 objective coefficient need not appear in objective line!
            objective_.resize(var_index+1,0.0);
        var_permutation_.push_back(var_permutation_.size()-1);
        return var_index;
    }

    bool ILP_input::var_exists(const std::string& var) const
    {
        return var_name_to_index_.count(var) > 0;
    }

    size_t ILP_input::get_var_index(const std::string& var) const
    {
        assert(var_exists(var));
        return var_name_to_index_.find(var)->second;
    }

    std::string ILP_input::get_var_name(const size_t index) const
    {
        assert(index < var_index_to_name_.size());
        return var_index_to_name_[index];
    }

    size_t ILP_input::get_or_create_variable_index(const std::string& var)
    {
        if(var_exists(var))
            return get_var_index(var);
        else 
            return add_new_variable(var); 
    }

    size_t ILP_input::nr_variables() const
    {
        assert(var_index_to_name_.size() == objective_.size());
        return var_index_to_name_.size();
    }

    void ILP_input::add_to_objective(const double coefficient, const std::string& var)
    {
        add_to_objective(coefficient, get_or_create_variable_index(var));
    }

    void ILP_input::add_to_objective(const double coefficient, const size_t var)
    {
        assert(var < objective_.size());
        objective_[var] += coefficient;
    }

    double ILP_input::objective(const size_t var) const
    {
        if(var >= nr_variables())
        {
            std::cout << var << ", " << nr_variables() << "\n";
            throw std::runtime_error("variable not present");
        }
        if(var >= objective_.size())
            return 0.0;
        return objective_[var];
    }

    double ILP_input::objective(const std::string& var) const
    {
        return objective(get_var_index(var));
    }

    size_t ILP_input::begin_new_inequality()
    {
        constraints_.push_back({});
        // set defaul inequality identifier
        std::string ineq_name = "INEQ_NR_" + std::to_string(constraints_.size()-1);
        set_inequality_identifier(ineq_name);
        return constraints_.size()-1;
    }

    void ILP_input::set_inequality_identifier(const std::string& identifier)
    {
        assert(constraints_.size() > 0);
        if(identifier == "")
            return;
        const std::string& old_identifier = constraints_.back().identifier;
        if(old_identifier != "")
        {
            assert(inequality_identifier_to_index_.count(old_identifier) > 0);
            inequality_identifier_to_index_.erase(old_identifier);
        }
        constraints_.back().identifier = identifier;
        if(inequality_identifier_to_index_.count(identifier) > 0)
            throw std::runtime_error("Duplicate inequality identifier " + identifier);
        inequality_identifier_to_index_.insert({identifier, constraints_.size()-1});
    }

    void ILP_input::set_inequality_type(const inequality_type ineq)
    {
        assert(constraints_.size() > 0);
        constraints_.back().ineq = ineq;
    }

    size_t ILP_input::add_constraint(const std::vector<int>& coefficients, const std::vector<size_t>& vars, const ILP_input::inequality_type ineq, const int right_hand_side)
    {
        assert(coefficients.size() == vars.size());
        for(const size_t var : vars)
        {
            if(var >= nr_variables())
            {
                for(size_t i=nr_variables(); i<=var; ++i)
                {
                    const std::string var_name = "x_" + std::to_string(i);
                    assert(!var_exists(var_name));
                    add_new_variable(var_name);
                }
            }
        }

        constraint constr;
        constr.coefficients = coefficients;
        constr.monomials = two_dim_variable_array<size_t>(std::vector<size_t>(vars.size(), 1));
        for(size_t i=0; i<vars.size(); ++i)
            constr.monomials(i,0) = vars[i];
        constr.ineq = ineq;
        constr.right_hand_side = right_hand_side;
        constraints_.push_back(constr);

        return constraints_.size()-1;
    }

    void ILP_input::add_to_constraint(const int coefficient, const size_t var)
    {
        assert(constraints_.size() > 0);
        auto& constr = constraints_.back();
        constr.coefficients.push_back(coefficient);
        std::array<size_t,1> vars{var};
        constr.monomials.push_back(vars.begin(), vars.end());
    }

    void ILP_input::add_to_constraint(const int coefficient, const std::string& var)
    {
        add_to_constraint(coefficient, get_or_create_variable_index(var));
    }

    void ILP_input::set_right_hand_side(const int x)
    {
        assert(constraints_.size() > 0);
        constraints_.back().right_hand_side = x;
    } 

    size_t ILP_input::nr_constraints() const
    {
        return constraints_.size();
    }

    bool ILP_input::preprocess()
    {
        std::vector<size_t> constraint_map;
        for (auto it = constraints_.begin(); it != constraints_.end(); it++)
        {
            bool remove = false;

            // empty constraints
            if (it->monomials.size() == 0)
            {
                // feasibility check
                switch (it->ineq)
                {
                    case inequality_type::smaller_equal:
                        if(it->right_hand_side < 0)
                            return false;
                        break;
                    case inequality_type::greater_equal:
                        if(it->right_hand_side > 0)
                            return false;
                        break;
                    case inequality_type::equal:
                        if(it->right_hand_side != 0)
                            return false;
                        break;
                    default:
                        throw std::runtime_error("inequality type not supported");
                }
                remove = true;
            }

            // variable bounds and fixations
            if (it->monomials.size() == 1 && it->monomials.size(0) == 1)
            {
                assert(it->monomials.size() == it->coefficients.size());
                const size_t variable = it->monomials(0,0);
                const int coeff = it->coefficients[0];
                switch (it->ineq)
                {
                    case inequality_type::smaller_equal:
                        if(std::min(coeff, 0) > it->right_hand_side)
                            return false;
                        if(std::max(coeff, 0) <= it->right_hand_side)
                            remove = true;
                        break;
                    case inequality_type::greater_equal:
                        if(std::max(coeff, 0) < it->right_hand_side)
                            return false;
                        if(std::min(coeff, 0) >= it->right_hand_side)
                            remove = true;
                        break;
                    case inequality_type::equal:
                        if(it->right_hand_side != 0 && it->right_hand_side != coeff)
                            return false;
                        break;
                    default:
                        throw std::runtime_error("inequality type not supported");
                }
            }

            // TODO implement preprocessing of variable/group fixations (e.g. x = 1, x + y = 0)

            // remove redundant constraint
            if (remove)
            {
                if(constraint_map.size() == 0)
                {
                    constraint_map.resize(constraints().size());
                    std::iota(constraint_map.begin(), constraint_map.end(), 0);
                }

                constraint_map[constraints_.size()-1] = std::distance(constraints_.begin(), it);
                constraint_map[ std::distance(constraints_.begin(), it) ] = std::numeric_limits<size_t>::max();

                *it = constraints_.back();
                constraints_.pop_back();
                it--;    
            }

            // remap constraint group indices
            if(constraint_map.size() > 0)
            {
                for(size_t cg=0; cg<coalesce_sets_.size(); ++cg)
                    for(size_t idx=0; idx<coalesce_sets_.size(cg); ++idx)
                        coalesce_sets_(cg,idx) = constraint_map[coalesce_sets_(cg,idx)];
            }
        }

        return true;
    }

    inline two_dim_variable_array<size_t> ILP_input::variable_adjacency_matrix() const
    {
        MEASURE_FUNCTION_EXECUTION_TIME
        std::vector<Eigen::Triplet<int>> var_constraint_adjacency_list;

        for(size_t i=0; i<this->constraints_.size(); ++i) 
        {
            const auto vars = variables(i);
            for(const size_t var : vars)
                var_constraint_adjacency_list.push_back({var, i, 1});
        }

        Eigen::SparseMatrix<int> A(this->nr_variables(), this->constraints_.size());
        A.setFromTriplets(var_constraint_adjacency_list.begin(), var_constraint_adjacency_list.end());
        const Eigen::SparseMatrix<int> adj_matrix = A*A.transpose();
        assert(adj_matrix.cols() == nr_variables() && adj_matrix.rows() == nr_variables());

        std::vector<size_t> adjacency_size(nr_variables(), 0);

        for(size_t i=0; i<adj_matrix.outerSize(); ++i) {
            for(typename Eigen::SparseMatrix<int>::InnerIterator it(adj_matrix,i); it; ++it) {
                if(it.value() > 0 && i != it.index()) {
                    adjacency_size[i]++;
                }
            }
        }

        two_dim_variable_array<size_t> adjacency(adjacency_size.begin(), adjacency_size.end());
        std::fill(adjacency_size.begin(), adjacency_size.end(), 0);

        for(size_t i=0; i<adj_matrix.outerSize(); ++i) {
            for(typename Eigen::SparseMatrix<int>::InnerIterator it(adj_matrix,i); it; ++it) {
                if(it.value() > 0 && i != it.index()) {
                    adjacency(i,adjacency_size[i]++) = it.index();
                }
            }
        }

        for(size_t i=0; i<adjacency.size(); ++i) {
            assert(std::is_sorted(adjacency[i].begin(), adjacency[i].end()));
        }

        return adjacency;
    }

    /*
       inline two_dim_variable_array<size_t> ILP_input::variable_adjacency_matrix() const
       {
    //std::unordered_set<std::array<size_t,2>> adjacent_vars;
    tsl::robin_set<std::array<size_t,2>> adjacent_vars;
    for(const auto& l : this->linear_constraints_) {
    for(size_t i=0; i<l.variables.size(); ++i) {
    for(size_t j=i+1; j<l.variables.size(); ++j) {
    const size_t var1 = l.variables[i].var;
    const size_t var2 = l.variables[j].var;
    adjacent_vars.insert({std::min(var1,var2), std::max(var1, var2)});
    }
    }
    }

    std::vector<size_t> adjacency_size(this->nr_variables(),0);
    for(const auto [i,j] : adjacent_vars) {
    ++adjacency_size[i];
    ++adjacency_size[j];
    }

    two_dim_variable_array<size_t> adjacency(adjacency_size.begin(), adjacency_size.end());
    std::fill(adjacency_size.begin(), adjacency_size.end(), 0);
    for(const auto e : adjacent_vars) {
    const auto [i,j] = e;
    assert(i<j);
    assert(adjacency_size[i] < adjacency[i].size());
    assert(adjacency_size[j] < adjacency[j].size());
    adjacency(i, adjacency_size[i]++) = j;
    adjacency(j, adjacency_size[j]++) = i;
    }

    for(size_t i=0; i<adjacency.size(); ++i)
    std::sort(adjacency[i].begin(), adjacency[i].end());

    return adjacency;
    }
    */

    inline two_dim_variable_array<size_t> ILP_input::bipartite_variable_bdd_adjacency_matrix() const
    {
        MEASURE_FUNCTION_EXECUTION_TIME
        std::vector<std::pair<size_t,size_t>> var_bdd_adj_list;
        std::vector<size_t> degrees(this->nr_variables() + this->nr_constraints(), 0);

        // determine list of adjacencies and vertex degrees
        for(size_t i=0; i<this->constraints_.size(); ++i) 
        {
            const auto vars = variables(i);
            for(const size_t var : vars)
            {
                var_bdd_adj_list.emplace_back(var, i);
                degrees[var]++;
                degrees[this->nr_variables() + i]++;
            }
        }

        // create adjacency matrix
        two_dim_variable_array<size_t> adjacency(degrees.begin(), degrees.end());
        std::fill(degrees.begin(), degrees.end(), 0);
        for (auto it = var_bdd_adj_list.begin(); it != var_bdd_adj_list.end(); it++)
        {
            size_t var = it->first;
            size_t bdd = this->nr_variables() + it->second;
            adjacency(var, degrees[var]++) = bdd;
            adjacency(bdd, degrees[bdd]++) = var;
        }

        return adjacency;
    }

    template<typename ZEROS_VAR_SET, typename ONES_VAR_SET>
    ILP_input ILP_input::reduce(const ZEROS_VAR_SET& zero_vars, const ONES_VAR_SET& one_vars) const
    {
        std::cout << "[ILP input] reduce problem with " << zero_vars.size() << " zero fixations and " << one_vars.size() << " one fixations\n";
        if(zero_vars.size() == 0 && one_vars.size() == 0)
            return *this;

        ILP_input reduced_ilp;
        std::vector<size_t> reduced_var_map;
        reduced_var_map.reserve(nr_variables());
        for(size_t i=0; i<nr_variables(); ++i)
        {
            assert(!(zero_vars.count(i) && one_vars.count(i)));
            if(!zero_vars.count(i) && !one_vars.count(i))
            {
                const double obj = objective(i);
                const size_t new_var = reduced_ilp.add_new_variable(get_var_name(i));
                reduced_ilp.add_to_objective(obj, new_var);
                reduced_var_map.push_back(new_var);
            }
            else
            {
                reduced_var_map.push_back(std::numeric_limits<size_t>::max());
                if(one_vars.count(i) > 0)
                    reduced_ilp.add_to_constant(objective(i));
            }
        }

        // set reduced constraints
        for(auto& constr : constraints_)
        {
            two_dim_variable_array<size_t> new_monomials;
            std::vector<int> new_coefficients;
            std::vector<size_t> unset_monomial_vars;
            std::vector<size_t> one_monomial_vars;
            std::vector<size_t> zero_monomial_vars;
            int new_right_hand_side = constr.right_hand_side;

            for(size_t monomial_ctr=0; monomial_ctr<constr.monomials.size(); ++monomial_ctr)
            {
                unset_monomial_vars.clear();
                one_monomial_vars.clear();
                zero_monomial_vars.clear();
                for(size_t var_ctr=0; var_ctr<constr.monomials.size(monomial_ctr); ++var_ctr)
                {
                    const size_t var = constr.monomials(monomial_ctr, var_ctr);
                    assert(zero_vars.count(var) == 0 || one_vars.count(var) == 0);
                    if(zero_vars.count(var) > 0)
                    {
                        assert(reduced_var_map[var] == std::numeric_limits<size_t>::max());
                        zero_monomial_vars.push_back(var);
                    }
                    else if(one_vars.count(var) > 0)
                    {
                        assert(reduced_var_map[var] == std::numeric_limits<size_t>::max());
                        one_monomial_vars.push_back(var);
                    }
                    else
                        unset_monomial_vars.push_back(reduced_var_map[var]);
                }
                if(zero_monomial_vars.size() > 0)
                {}
                else if(one_monomial_vars.size() == constr.monomials.size(monomial_ctr))
                {
                    new_right_hand_side -= constr.coefficients[monomial_ctr];
                }
                else
                {
                    new_monomials.push_back(unset_monomial_vars.begin(), unset_monomial_vars.end());
                    new_coefficients.push_back(constr.coefficients[monomial_ctr]);
                }
            }
            if(new_monomials.size() > 0)
            {
                ILP_input::constraint new_constr;
                new_constr.identifier = constr.identifier;
                new_constr.coefficients = new_coefficients;
                new_constr.monomials = new_monomials;
                new_constr.ineq = constr.ineq;
                new_constr.right_hand_side = new_right_hand_side;
                reduced_ilp.add_constraint(new_constr);
            }
            else
            {
                if(new_right_hand_side != 0 && constr.ineq == inequality_type::equal
                        || new_right_hand_side > 0 && constr.ineq == inequality_type::smaller_equal
                        || new_right_hand_side < 0 && constr.ineq == inequality_type::greater_equal)
                    throw std::runtime_error("reduced model not feasible due to violated constraint " + constr.identifier);
                continue;
            }
        }

        return reduced_ilp;
    }

    template ILP_input ILP_input::reduce(const std::unordered_set<size_t>&, const std::unordered_set<size_t>&) const;

    permutation ILP_input::reorder(ILP_input::variable_order var_ord)
    {
        if(var_ord == variable_order::input)
            var_permutation_ = permutation(nr_variables());
        else if(var_ord == variable_order::bfs)
            var_permutation_ = this->reorder_bfs();
        else if(var_ord == variable_order::cuthill)
            var_permutation_ = this->reorder_Cuthill_McKee();
        else
        {
            assert(var_ord == variable_order::mindegree);
            var_permutation_ = this->reorder_minimum_degree_ordering();
        }

        return var_permutation_;
    }

    void ILP_input::reorder(const permutation& order)
    {
        MEASURE_FUNCTION_EXECUTION_TIME;
        assert(order.size() == this->nr_variables());
        assert(order.is_permutation());
        std::vector<double> new_objective(this->nr_variables(), 0.0);
        std::vector<size_t> inverse_order(this->nr_variables());
        std::vector<std::string> new_var_index_to_name(this->nr_variables());

        for(size_t i=0; i<this->nr_variables(); ++i)
        {
            assert(var_name_to_index_.count(var_index_to_name_[order[i]]) > 0);
            var_name_to_index_.find(var_index_to_name_[order[i]]).value() = i;
        }

//#pragma omp parallel for schedule(guided)
        for(size_t i=0; i<this->nr_variables(); ++i)
        {
            if(order[i] < this->objective_.size())
                new_objective[i] = this->objective_[order[i]]; 

            if(order[i] < this->var_index_to_name_.size())
                new_var_index_to_name[i] = std::move(this->var_index_to_name_[order[i]]);

            inverse_order[order[i]] = i;

        }
        std::swap(this->objective_, new_objective);
        std::swap(new_var_index_to_name, this->var_index_to_name_);

//#pragma omp parallel for schedule(guided)
        for(size_t ineq_nr=0; ineq_nr<this->constraints_.size(); ++ineq_nr)
        {
            auto& constr = this->constraints_[ineq_nr];
            for(size_t monomial_idx=0; monomial_idx<constr.monomials.size(); ++monomial_idx)
            {
                for(size_t var_idx=0; var_idx<constr.monomials.size(monomial_idx); ++var_idx)
                {
                    auto& var = constr.monomials(monomial_idx, var_idx);
                    var = inverse_order[var];
                }
            }
        }
    }

    inline permutation ILP_input::reorder_bfs()
    {
        const auto adj = bipartite_variable_bdd_adjacency_matrix();
        const auto order = bfs_ordering(adj, this->nr_variables());
        reorder(order);
        return order;
    }

    inline permutation ILP_input::reorder_Cuthill_McKee()
    {
        const auto adj = bipartite_variable_bdd_adjacency_matrix();
        const auto order = Cuthill_McKee(adj, this->nr_variables());
        reorder(order);
        return order;
    }

    inline permutation ILP_input::reorder_minimum_degree_ordering()
    {
        const auto adj = bipartite_variable_bdd_adjacency_matrix();
        const auto order = minimum_degree_ordering(adj, this->nr_variables());
        reorder(order);
        return order;
    }

    size_t ILP_input::nr_constraint_groups() const
    {
        return coalesce_sets_.size(); 
    }

    std::tuple<const size_t*, const size_t*> ILP_input::constraint_group(const size_t i) const
    {
        assert(i < nr_constraint_groups());
        return std::make_tuple(coalesce_sets_[i].begin(), coalesce_sets_[i].end());
    }

    std::tuple<Eigen::SparseMatrix<int>, Eigen::MatrixXi> ILP_input::export_constraints() const
    {
        using T = Eigen::Triplet<int>;
        std::vector<T> coefficients;

        for(size_t c=0; c<nr_constraints(); ++c)
        {
            for(size_t monomial_idx=0; monomial_idx<constraints()[c].monomials.size(); ++monomial_idx)
            {
                if(constraints()[c].monomials.size(monomial_idx) != 1)
                    throw std::runtime_error("instance has higher order constraints, cannot export to matrix");
                const size_t var = constraints()[c].monomials(monomial_idx, 0);
                const int coeff = constraints()[c].coefficients[monomial_idx];
                coefficients.push_back(T(c, var, coeff));

            }
        }

        Eigen::SparseMatrix<int> A(nr_constraints(), nr_variables());
        A.setFromTriplets(coefficients.begin(), coefficients.end());

        Eigen::MatrixXi b(nr_constraints(), 2);
        for(size_t c=0; c<nr_constraints(); ++c)
        {
            if(constraints()[c].ineq == inequality_type::equal)
            {
                b(c,0) = constraints()[c].right_hand_side;
                b(c,1) = constraints()[c].right_hand_side;
            }
            else if(constraints()[c].ineq == inequality_type::smaller_equal)
            {
                b(c,0) = std::numeric_limits<int>::min();
                b(c,1) = constraints()[c].right_hand_side;
            }
            else if(constraints()[c].ineq == inequality_type::greater_equal)
            {
                b(c,0) = constraints()[c].right_hand_side;
                b(c,1) = std::numeric_limits<int>::max();
            }
            else
            {
                assert(false);
            } 
        }

        return {A,b};
    }

    bool ILP_input::every_variable_in_some_ineq() const
    {
        std::vector<size_t> nr_ineqs_per_var(nr_variables(), 0);
        for(const auto& c : constraints_)
        {
            for(size_t monomial_idx=0; monomial_idx<c.monomials.size(); ++monomial_idx)
            {
                for(size_t var_idx=0; var_idx<c.monomials.size(monomial_idx); ++var_idx)
                {
                    const size_t& var = c.monomials(monomial_idx, var_idx);
                    ++nr_ineqs_per_var[var];
                }
            }
        }

        return std::count(nr_ineqs_per_var.begin(), nr_ineqs_per_var.end(), 0) == 0;
    }

    size_t ILP_input::nr_disconnected_subproblems() const
    {
        union_find uf(nr_variables() + nr_constraints());
        for(size_t c=0; c<nr_constraints(); ++c)
        {
            for(size_t monomial_idx=0; monomial_idx<constraints_[c].monomials.size(); ++monomial_idx)
            {
                for(size_t var_idx=0; var_idx<constraints_[c].monomials.size(monomial_idx); ++var_idx)
                {
                    const size_t& var = constraints_[c].monomials(monomial_idx, var_idx);
                    uf.merge(var, nr_variables() + c);
                }
            }
        }

        return uf.count();
    }

}
