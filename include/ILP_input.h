#pragma once

#include <vector>
#include <string>
#include <cassert>
#include <algorithm>
#include "two_dimensional_variable_array.hxx"
#include "permutation.hxx"
#include <tsl/robin_map.h>
#include <tsl/robin_set.h>
#include <Eigen/SparseCore>

namespace LPMP {

    class ILP_input {
        public:

        enum class inequality_type { smaller_equal, greater_equal, equal };

        // can be linear or polynomial constraint
        struct constraint {
            std::string identifier;
            std::vector<int> coefficients;
            two_dim_variable_array<size_t> monomials;
            inequality_type ineq;
            int right_hand_side;
            void normalize();
            bool is_normalized() const;
            bool is_linear() const;
            bool is_simplex() const;
            // if constraint is non-linear check if all variables appear at most once
            bool distinct_variables() const;
            static bool monomials_cmp(const two_dim_variable_array<size_t>& monomials, const size_t idx1, const size_t idx2);
        };

        enum class variable_order { input, bfs, cuthill, mindegree } variable_order_{variable_order::input};

        std::vector<size_t> variables(const size_t ineq_nr) const;
        bool var_exists(const std::string& var) const;
        size_t get_var_index(const std::string& var) const;
        std::string get_var_name(const size_t index) const;
        size_t add_new_variable(const std::string& var);
        size_t get_or_create_variable_index(const std::string& var);
        size_t nr_variables() const;
        void add_to_objective(const double coefficient, const std::string& var);
        void add_to_objective(const double coefficient, const size_t var);
        const std::vector<double>& objective() const { return objective_; };
        double objective(const size_t var) const;
        double objective(const std::string& var) const;
        size_t begin_new_inequality();
        void set_inequality_identifier(const std::string& identifier);
        void set_inequality_type(const inequality_type ineq);
        size_t add_constraint(const std::vector<int>& coefficients, const std::vector<size_t>& vars, const inequality_type ineq, const int right_hand_side);
        size_t add_constraint(const constraint& c) { constraints_.push_back(c); return constraints_.size()-1; }
        void add_to_constraint(const int coefficient, const size_t var);
        void add_to_constraint(const int coefficient, const std::string& var);
        // add monomial to constraint
        template<typename ITERATOR>
            void add_to_constraint(const int coefficient, ITERATOR var_idx_begin, ITERATOR var_idx_end);
        void set_right_hand_side(const int x);
        size_t nr_constraints() const;
        const auto& constraints() const { return constraints_; }

        template<typename ITERATOR>
            bool feasible(ITERATOR begin, ITERATOR end) const;

        template<typename ITERATOR>
            double evaluate(ITERATOR begin, ITERATOR end) const; 

        template<typename STREAM>
            void write_lp(STREAM& s, const constraint & constr) const;
        template<typename STREAM>
            void write_lp(STREAM& s) const;

        template<typename STREAM>
            void write_opb(STREAM& s, const constraint & constr) const;
        template<typename STREAM>
            void write_opb(STREAM& s) const;

        bool preprocess();
        void normalize();
        bool is_normalized() const;

        template<typename ZEROS_VAR_SET, typename ONES_VAR_SET>
            ILP_input reduce(const ZEROS_VAR_SET& zero_vars, const ONES_VAR_SET& one_vars) const;

        permutation reorder(variable_order var_ord);
        permutation reorder_bfs();
        permutation reorder_Cuthill_McKee(); 
        permutation reorder_minimum_degree_ordering();
        void reorder(const permutation& new_order);
        permutation get_variable_permutation() const { return var_permutation_; }

        template<typename ITERATOR>
            size_t add_constraint_group(ITERATOR begin, ITERATOR end);

        size_t nr_constraint_groups() const;
        std::tuple<const size_t*, const size_t*> constraint_group(const size_t i) const;

        std::tuple<Eigen::SparseMatrix<int>, Eigen::MatrixXi> export_constraints() const;
        const std::vector<std::string>& var_index_to_name() const { return var_index_to_name_; };
        const tsl::robin_map<std::string, size_t>& var_name_to_index() const { return var_name_to_index_; };
        const tsl::robin_map<std::string, size_t>& inequality_identifier_to_index() const { return inequality_identifier_to_index_; };

        void add_to_constant(const double x) { constant_ += x; }
        double constant() const { return constant_; }

        // sanity checks on instance
        // is every variable contained in at least one constraint?
        bool every_variable_in_some_ineq() const;
        // does variable/constraint matrix form a single connected component or can problem be decomposed
        size_t nr_disconnected_subproblems() const;

        private:

            std::vector<constraint> constraints_;
            std::vector<double> objective_;
            double constant_ = 0.0;
            std::vector<std::string> var_index_to_name_;
            tsl::robin_map<std::string, size_t> var_name_to_index_;
            tsl::robin_map<std::string, size_t> inequality_identifier_to_index_;
            two_dim_variable_array<size_t> coalesce_sets_;

            permutation var_permutation_;

        private:
            two_dim_variable_array<size_t> variable_adjacency_matrix() const;
            two_dim_variable_array<size_t> bipartite_variable_bdd_adjacency_matrix() const;
    };

    template<typename ITERATOR>
        void ILP_input::add_to_constraint(const int coefficient, ITERATOR var_idx_begin, ITERATOR var_idx_end)
        {
            assert(constraints_.size() > 0);
            assert(std::distance(var_idx_begin, var_idx_end) > 0);
            assert(*std::max_element(var_idx_begin, var_idx_end) < nr_variables());
            auto& constr = constraints_.back();
            assert(constr.coefficients.size() == constr.monomials.size());
            constr.coefficients.push_back(coefficient);
            constr.monomials.push_back(var_idx_begin, var_idx_end);
        }

    template<typename ITERATOR>
        bool ILP_input::feasible(ITERATOR begin, ITERATOR end) const
        {
            if(std::distance(begin, end) != nr_variables())
                return false;

            for(const auto& l : constraints_) {
                int s = 0;
                for(size_t monomial_idx=0; monomial_idx<l.monomials.size(); ++monomial_idx)
                {
                    int val = 1;
                    for(size_t var_idx=0; var_idx<l.monomials.size(monomial_idx); ++var_idx)
                    {
                        const size_t var = l.monomials(monomial_idx,var_idx);
                        const int var_val = *(begin + var);
                        assert(var_val == 0 || var_val == 1);
                        val *= var_val;
                    }
                    s += l.coefficients[monomial_idx] * val;
                }
                switch(l.ineq) {
                    case inequality_type::smaller_equal:
                        if(s > l.right_hand_side)
                            return false;
                        break;
                    case inequality_type::greater_equal:
                        if(s < l.right_hand_side)
                            return false;
                        break;
                    case inequality_type::equal:
                        if(s != l.right_hand_side)
                            return false;
                        break;
                    default:
                        throw std::runtime_error("inequality type not supported");
                }
            }
            return true;
        }

    template<typename ITERATOR>
        double ILP_input::evaluate(ITERATOR begin, ITERATOR end) const
        {
            if(!feasible(begin,end))
                return std::numeric_limits<double>::infinity();
            assert(std::distance(begin,end) >= objective_.size());
            double cost = constant_;
            for(size_t i=0; i<objective_.size(); ++i) {
                assert(*(begin+i) == 0 || *(begin+i) == 1);
                cost += objective_[i] * *(begin+i);
            }
            return cost;
        }

        template<typename ITERATOR>
            size_t ILP_input::add_constraint_group(ITERATOR begin, ITERATOR end)
            {
                if constexpr(std::is_integral_v<std::remove_reference_t<decltype(*begin)>>)
                {
                    for(auto it=begin; it!=end; ++it)
                        assert(*it < constraints().size());
                    coalesce_sets_.push_back(begin, end); 
                }
                else if constexpr(std::is_convertible_v<std::remove_reference_t<decltype(*begin)>, std::string>)
                {
                    std::vector<size_t> lineq_nrs;
                    for(auto it=begin; it!=end; ++it)
                    {
                        auto ineq_nr_it = inequality_identifier_to_index_.find(*it);
                        if(ineq_nr_it == inequality_identifier_to_index_.end())
                            throw std::runtime_error("inequality identifier " + *it + " not present");
                        assert(ineq_nr_it->second < constraints().size());
                        lineq_nrs.push_back(ineq_nr_it->second);
                    }
                    coalesce_sets_.push_back(lineq_nrs.begin(), lineq_nrs.end()); 
                } else {
                    static_assert("add_constraint_group must be called with iterators referencing integral types or std::string");
                }

                return coalesce_sets_.size()-1;
            }

    template<typename STREAM>
        void ILP_input::write_lp(STREAM& s, const constraint & constr) const
        {
            assert(constr.coefficients.size() == constr.monomials.size());
            s << constr.identifier << ": ";
            for(size_t monomial_idx=0; monomial_idx<constr.coefficients.size(); ++monomial_idx)
            {
                const double coeff = constr.coefficients[monomial_idx];
                if(coeff != 0.0)
                {
                    assert(constr.monomials.size(monomial_idx) > 0);
                    s << (coeff < 0.0 ? " - " : " + ") <<  std::abs(coeff);
                    for(size_t var_idx=0; var_idx<constr.monomials.size(monomial_idx); ++var_idx)
                    {
                        const size_t var = constr.monomials(monomial_idx, var_idx);
                        assert(var < var_index_to_name_.size());
                        s << " " << var_index_to_name_[var];
                    }
                }
            }

            switch(constr.ineq) {
                case inequality_type::smaller_equal:
                    s << " <= ";
                    break;
                case inequality_type::greater_equal:
                    s << " >= ";
                    break;
                case inequality_type::equal:
                    s << " = ";
                    break;
                default:
                    throw std::runtime_error("inequality type not supported");
                    break;
            }
            s << constr.right_hand_side << "\n";   
        }

    template<typename STREAM>
        void ILP_input::write_lp(STREAM& s) const
        {
            s << "Minimize\n";
            for(size_t i=0; i<objective_.size(); ++i)
                s << (objective_[i] < 0.0 ? " - " : " + ") <<  std::abs(objective_[i]) << " " << var_index_to_name_[i] << "\n";
            if(constant_ != 0.0)
                s << " + " << constant_ << "\n";
            s << "Subject To\n";
            for(const auto& constr : constraints()) {
                write_lp(s, constr);
            }
            if(coalesce_sets_.size() > 0)
            {
                s << "Coalesce\n";
                for(size_t c=0; c<coalesce_sets_.size(); ++c)
                {
                    for(size_t j=0; j<coalesce_sets_.size(c); ++j)
                    {
                        const size_t ineq_nr = coalesce_sets_(c,j);
                        assert(ineq_nr < constraints_.size());
                        const std::string inequality_id = constraints_[ineq_nr].identifier;
                        assert(inequality_id != "");
                        s << inequality_id << " ";
                    }
                    s << "\n";
                }
            }
            s << "Bounds\n";
            s << "Binaries\n";
            for(const auto& v : var_index_to_name_)
                s << v << "\n";
            s << "End\n";
        } 

    template<typename STREAM>
        void ILP_input::write_opb(STREAM& s) const
        {
            s << "* #variable= " << nr_variables() << " #constraint= " << nr_constraints() << "\n";

            s << "min:";
            for(size_t i=0; i<nr_variables(); ++i)
                s << " " << (objective(i) < 0.0 ? " -" : " +") <<  std::abs(objective(i)) << " " << "x" << i+1; 
            s << ";\n";
            for(const auto& constr : constraints())
                write_opb(s, constr);
        }

    template<typename STREAM>
        void ILP_input::write_opb(STREAM& s, const constraint & constr) const
        {
            // OPB only accepts = and >= constraints
            const double coeff_multiplier = [&]() {;
                if(constr.ineq == inequality_type::smaller_equal)
                    return -1.0;
                else
                    return 1.0;
            }();

            assert(constr.monomials.size() == constr.coefficients.size());
            for(size_t monomial_idx=0; monomial_idx<constr.monomials.size(); ++monomial_idx)
            {
                const double coeff = constr.coefficients[monomial_idx];
                if(coeff != 0)
                {
                    assert(constr.monomials.size(monomial_idx) > 0);
                    s << ((coeff_multiplier * coeff) < 0.0 ? " -" : " +") <<  std::abs(coeff);
                    for(size_t var_idx=0; var_idx<constr.monomials.size(monomial_idx); ++var_idx)
                    {
                        const size_t var = constr.monomials(monomial_idx, var_idx);
                        assert(var < var_index_to_name_.size());
                        s << " x" << var+1;
                    }
                }
            }

            switch(constr.ineq) {
                case inequality_type::smaller_equal:
                    s << " >= ";
                    break;
                case inequality_type::greater_equal:
                    s << " >= ";
                    break;
                case inequality_type::equal:
                    s << " = ";
                    break;
                default:
                    throw std::runtime_error("inequality type not supported");
                    break;
            }
            s << coeff_multiplier * constr.right_hand_side << ";\n";   
        }
}
