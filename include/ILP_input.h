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
        struct weighted_variable {
            int coefficient;
            size_t var;
            bool operator<(const weighted_variable& o) const { return var < o.var; }
        };

        enum class inequality_type { smaller_equal, greater_equal, equal };

        struct linear_constraint {
            std::string identifier;
            std::vector<weighted_variable> variables;
            inequality_type ineq;
            int right_hand_side;
            void normalize() { std::sort(variables.begin(), variables.end()); }
        };

        enum class variable_order { input, bfs, cuthill, mindegree } variable_order_{variable_order::input};

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
        void add_to_constraint(const int coefficient, const size_t var);
        void add_to_constraint(const int coefficient, const std::string& var);
        void set_right_hand_side(const int x);
        size_t nr_constraints() const;
        const auto& constraints() const { return linear_constraints_; }

        template<typename ITERATOR>
            bool feasible(ITERATOR begin, ITERATOR end) const;

        template<typename ITERATOR>
            double evaluate(ITERATOR begin, ITERATOR end) const; 

        template<typename STREAM>
            void write_lp(STREAM& s, const linear_constraint & constr) const;
        template<typename STREAM>
            void write_lp(STREAM& s) const;

        template<typename STREAM>
            void write_opb(STREAM& s, const linear_constraint & constr) const;
        template<typename STREAM>
            void write_opb(STREAM& s) const;

        bool preprocess();

        permutation reorder(variable_order var_ord);
        permutation reorder_bfs();
        permutation reorder_Cuthill_McKee(); 
        permutation reorder_minimum_degree_ordering();
        void reorder(const permutation& new_order);

        template<typename ITERATOR>
            void add_constraint_group(ITERATOR begin, ITERATOR end);

        size_t nr_constraint_groups() const;
        std::tuple<const size_t*, const size_t*> constraint_group(const size_t i) const;

        std::tuple<Eigen::SparseMatrix<int>, Eigen::MatrixXi> export_constraints() const;

        private:
            std::vector<linear_constraint> linear_constraints_;
            std::vector<double> objective_;
            std::vector<std::string> var_index_to_name_;
            tsl::robin_map<std::string, size_t> var_name_to_index_;
            tsl::robin_map<std::string, size_t> inequality_identifier_to_index_;
            two_dim_variable_array<size_t> coalesce_sets_;

        private:
            two_dim_variable_array<size_t> variable_adjacency_matrix() const;
            two_dim_variable_array<size_t> bipartite_variable_bdd_adjacency_matrix() const;
    };

    template<typename ITERATOR>
        bool ILP_input::feasible(ITERATOR begin, ITERATOR end) const
        {
            if(std::distance(begin, end) != nr_variables())
                return false;

            for(const auto& l : linear_constraints_) {
                int s = 0;
                for(const auto v : l.variables) {
                    assert(*(begin + v.var) == 0 || *(begin + v.var) == 1);
                    s += v.coefficient * *(begin + v.var);
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
            double cost = 0.0;
            for(size_t i=0; i<objective_.size(); ++i) {
                assert(*(begin+i) == 0 || *(begin+i) == 1);
                cost += objective_[i] * *(begin+i);
            }
            return cost;
        }

        template<typename ITERATOR>
            void ILP_input::add_constraint_group(ITERATOR begin, ITERATOR end)
            {
                if constexpr(std::is_integral_v<decltype(*begin)>)
                {
                    coalesce_sets_.push_back(begin, end); 
                }
                else if constexpr(std::is_convertible_v<decltype(*begin), std::string>)
                {
                    std::vector<size_t> lineq_nrs;
                    for(auto it=begin; it!=end; ++it)
                    {
                        auto ineq_nr_it = inequality_identifier_to_index_.find(*it);
                        if(ineq_nr_it == inequality_identifier_to_index_.end())
                            throw std::runtime_error("inequality identifier " + *it + " not present");
                        lineq_nrs.push_back(ineq_nr_it->second);
                    }
                    coalesce_sets_.push_back(lineq_nrs.begin(), lineq_nrs.end()); 
                }
            }

    template<typename STREAM>
        void ILP_input::write_lp(STREAM& s, const linear_constraint & constr) const
        {
            for(const auto term : constr.variables) {
                if(term.coefficient != 0.0)
                {
                    assert(term.var < var_index_to_name_.size());
                    s << (term.coefficient < 0.0 ? "- " : "+ ") <<  std::abs(term.coefficient) << " " << var_index_to_name_[term.var] << " "; 
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
            for(const auto o : var_name_to_index_) {
                s << (objective(o.second) < 0.0 ? "- " : "+ ") <<  std::abs(objective(o.second)) << " " << o.first << "\n"; 
            }
            s << "Subject To\n";
            for(const auto& constr : constraints()) {
                write_lp(s, constr);
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
            s << "min:";
            for(const auto o : var_name_to_index_)
                s << " " << (objective(o.second) < 0.0 ? "- " : "+ ") <<  std::abs(objective(o.second)) << " " << o.first; 
            s << ";\n";
            for(const auto& constr : constraints())
                write_opb(s, constr);
        }

    template<typename STREAM>
        void ILP_input::write_opb(STREAM& s, const linear_constraint & constr) const
        {
            // OPB only accepts = and >= constraints
            const double coeff_multiplier = [&]() {;
                if(constr.ineq == inequality_type::smaller_equal)
                    return -1.0;
                else
                    return 1.0;
            }();

            for(const auto term : constr.variables) {
                if(term.coefficient != 0.0)
                {
                    assert(term.var < var_index_to_name_.size());
                    s << ((coeff_multiplier * term.coefficient) < 0.0 ? "- " : "+ ") <<  std::abs(term.coefficient) << " " << var_index_to_name_[term.var] << " "; 
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
