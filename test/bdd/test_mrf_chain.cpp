#include "bdd_manager/bdd_mgr.h"
#include "../test.h"
#include <vector>
#include <array>
#include <iostream> // TODO delete
#include <unistd.h>
#include <unordered_map>

using namespace BDD;
using namespace LPMP;

template<typename ITERATOR>
node_ref create_simplex(bdd_mgr& mgr, ITERATOR var_begin, ITERATOR var_end)
{
    std::vector<node_ref> nodes;
    nodes.push_back(mgr.or_rec(var_begin, var_end));
    for(size_t i=0; i<std::distance(var_begin, var_end); ++i) {
        for(size_t j=i+1; j<std::distance(var_begin, var_end); ++j) {
            nodes.push_back(mgr.or_rec(mgr.negate(*(var_begin+i)), mgr.negate(*(var_begin+j))));
        }
    }
    return mgr.and_rec(nodes.begin(), nodes.end());
}

template<typename ITERATOR>
node_ref create_marginalization_constraint(bdd_mgr& mgr, node_ref u, ITERATOR var_begin, ITERATOR var_end)
{
    std::vector<node_ref> bdds;
    std::vector<node_ref> all_vars({mgr.negate(u)});
    for(auto it=var_begin; it!=var_end; ++it)
    {
        bdds.push_back(mgr.or_rec(mgr.negate(*it), u));
        all_vars.push_back(*it);
    }
    bdds.push_back(mgr.or_rec(all_vars.begin(), all_vars.end()));
    return mgr.and_rec(bdds.begin(), bdds.end());
}

node_ref create_mrf_chain_2(bdd_mgr& mgr, const size_t nr_labels)
{
    std::vector<node_ref> u_vars_1;
    for(size_t i=0; i<nr_labels; ++i)
        u_vars_1.push_back(mgr.projection(i));
    node_ref us1 = create_simplex(mgr, u_vars_1.begin(), u_vars_1.end());

    std::vector<node_ref> p_vars;
    for(size_t i=0; i<nr_labels; ++i)
        for(size_t j=0; j<nr_labels; ++j)
            p_vars.push_back(mgr.projection(nr_labels + i*nr_labels + j));
    node_ref ps = create_simplex(mgr, p_vars.begin(), p_vars.end());

    std::vector<node_ref> u_vars_2;
    for(size_t i=0; i<nr_labels; ++i)
        u_vars_2.push_back(mgr.projection(nr_labels + nr_labels*nr_labels + i));
    node_ref us2 = create_simplex(mgr, u_vars_2.begin(), u_vars_2.end());

    std::vector<node_ref> marg_constrs;
    for(size_t i = 0; i<nr_labels; ++i)
    {
        std::vector<node_ref> p;
        for(size_t j = 0; j<nr_labels; ++j)
            p.push_back(p_vars[i*nr_labels + j]);
        marg_constrs.push_back(create_marginalization_constraint(mgr, u_vars_1[i], p.begin(), p.end()));
        ps = ps & marg_constrs.back();
    }

    for(size_t j = 0; j<nr_labels; ++j)
    {
        std::vector<node_ref> p;
        for(size_t i = 0; i<nr_labels; ++i)
            p.push_back(p_vars[i*nr_labels + j]);
        marg_constrs.push_back(create_marginalization_constraint(mgr, u_vars_2[j], p.begin(), p.end()));
        ps = ps & marg_constrs.back();
    }

    return us1 & us2 & ps;
}

node_ref create_mrf_chain_rec(bdd_mgr& mgr, const size_t nr_vars, const size_t nr_labels)
{
    assert(nr_vars >= 2);
    if(nr_vars == 2)
        return create_mrf_chain_2(mgr, nr_labels);
    else
    {
        assert(nr_vars % 2 == 0);
        node_ref mrf_chain_1 = create_mrf_chain_rec(mgr, nr_vars/2, nr_labels);

        std::unordered_map<size_t, size_t> m;
        const size_t n = mrf_chain_1.variables().size() + nr_labels * nr_labels;
        for(size_t i=0; i<n; ++i)
            m.insert({i,i+n});
        node_ref mrf_chain_2 = mgr.rebase(mrf_chain_1, m);

        node_ref mrf_chain_intermed = create_mrf_chain_2(mgr, nr_labels);
        m.clear();
        const size_t n2 = mrf_chain_intermed.variables().size();
        const size_t intermed_offset = n - nr_labels - nr_labels*nr_labels;
        for(size_t i=0; i<n2; ++i)
            m.insert({i,i+intermed_offset});
        node_ref mrf_chain_intermed_middle = mgr.rebase(mrf_chain_intermed, m);

        return mrf_chain_1 & mrf_chain_2 & mrf_chain_intermed_middle;
    }

}

node_ref create_mrf_chain(bdd_mgr& mgr, const size_t nr_vars, const size_t nr_labels)
{
    std::vector<node_ref> bdds;

    // create unary and pairwise variables interleaved
    std::vector<std::vector<node_ref>> unary_vars;
    std::vector<std::vector<std::vector<node_ref>>> pairwise_vars;
    size_t var_counter = 0;
    for(size_t i=0; i<nr_vars; ++i)
    {
        unary_vars.push_back({});
        for(size_t j = 0; j<nr_labels; ++j)
        {
            unary_vars.back().push_back(mgr.projection(var_counter++));
        }
        bdds.push_back(create_simplex(mgr,unary_vars.back().begin(), unary_vars.back().end()));

        pairwise_vars.push_back({});
        for(size_t l1 = 0; l1<nr_labels; ++l1)
        {
            pairwise_vars.back().push_back({});
            for(size_t l2 = 0; l2<nr_labels; ++l2)
            {
                pairwise_vars.back().back().push_back(mgr.projection(var_counter++));
            }
        }
    }

    auto pairwise_var = [&](const std::array<size_t,2> v1, const std::array<size_t,2> v2) {
        const size_t offset = nr_vars * nr_labels;
        assert(v1[0] +1 == v2[0]);
        const size_t pairwise_offset = v1[0] * nr_labels * nr_labels;
        const size_t label_idx = v1[1]*nr_labels + v2[1]; 
        return offset + pairwise_offset + label_idx; 
    };

    for(size_t i=0; i+1<nr_vars; ++i)
    {
        for(size_t l1 = 0; l1<nr_labels; ++l1)
        {
            std::vector<node_ref> p;
            for(size_t l2 = 0; l2<nr_labels; ++l2)
                p.push_back(pairwise_vars[i][l1][l2]);
            bdds.push_back(create_marginalization_constraint(mgr, unary_vars[i][l1], p.begin(), p.end()));
        }

        for(size_t l2 = 0; l2<nr_labels; ++l2)
        {
            std::vector<node_ref> p;
            for(size_t l1 = 0; l1<nr_labels; ++l1)
                p.push_back(pairwise_vars[i][l1][l2]);
            bdds.push_back(create_marginalization_constraint(mgr, unary_vars[i+1][l2], p.begin(), p.end()));
        }
    }
    
    node_ref c = mgr.and_rec(bdds.begin(), bdds.end());
    return c;
    
    //std::cout << "nr mrf nodes: " << c.nr_nodes() << "\n";
    //test(c.nr_nodes() == 47957, "nr of nodes for mrf chain not matching."); // nr valid for 1000 nodes and 3 labels
    //c->print(std::cout);
}

void test_mrf_chin_construction_equiv(bdd_mgr& mgr, const size_t nr_vars, const size_t nr_labels)
{
    node_ref mrf_rec = create_mrf_chain_rec(mgr, nr_vars, nr_labels);
    node_ref mrf_iter = create_mrf_chain(mgr, nr_vars, nr_labels);
    const auto v_rec = mrf_rec.variables();
    const auto v_iter = mrf_iter.variables();
    test(v_rec == v_iter, "mrf supports do not match");
    test(mrf_rec == mrf_iter, "mrf chain constructions results in different bdds)");

    std::cout << "nr mrf variables = " << nr_vars << ", nr labels = " << nr_labels << ", nr bdd variables = " << mrf_rec.variables().size() << ", nr bdd nodes = " << mrf_rec.nr_nodes() << ", ratio = " << mrf_rec.variables().size() / double(mrf_rec.nr_nodes()) << "\n";
}

int main(int argc, char** argv)
{
    bdd_mgr mgr;

    test_mrf_chin_construction_equiv(mgr, 8, 5);
    test_mrf_chin_construction_equiv(mgr, 8, 7);
    test_mrf_chin_construction_equiv(mgr, 8, 8);
    //test_mrf_chin_construction_equiv(mgr, 8, 9);
    //test_mrf_chin_construction_equiv(mgr, 8, 10);

    test_mrf_chin_construction_equiv(mgr, 2, 2);
    test_mrf_chin_construction_equiv(mgr, 2, 3);
    test_mrf_chin_construction_equiv(mgr, 2, 4);
    test_mrf_chin_construction_equiv(mgr, 2, 5);
    test_mrf_chin_construction_equiv(mgr, 2, 6);
    //test_mrf_chin_construction_equiv(mgr, 2, 7);
    //test_mrf_chin_construction_equiv(mgr, 2, 8);

    test_mrf_chin_construction_equiv(mgr, 4, 2);
    test_mrf_chin_construction_equiv(mgr, 4, 3);
    test_mrf_chin_construction_equiv(mgr, 4, 4);

    node_ref mrf_1024_3 = create_mrf_chain_rec(mgr, 1024, 3);
    //node_ref mrf_8_10 = create_mrf_chain_rec(mgr, 8, 10);

    const size_t nr_labels = 10;
    node_ref bdd = create_mrf_chain_rec(mgr, 2, nr_labels);
    for(size_t i=0; i<nr_labels; ++i)
    {
        for(size_t j=0; j<nr_labels; ++j)
        {
            std::vector<char> l(bdd.variables().size(), 0);
            l[i] = 1;
            l[nr_labels + i*nr_labels + j] = 1;
            l[nr_labels + nr_labels*nr_labels + j] = 1;
            test(bdd.evaluate(l.begin(), l.end()) == true, " mrf chain labeling error.");

            if(i != j)
            {
                std::fill(l.begin(), l.end(), 0);
                l[i] = 1;
                l[nr_labels + j*nr_labels + i] = 1;
                l[nr_labels + nr_labels*nr_labels + j] = 1;
                test(bdd.evaluate(l.begin(), l.end()) == false, " mrf chain labeling error.");
            }
        }
    }
}
