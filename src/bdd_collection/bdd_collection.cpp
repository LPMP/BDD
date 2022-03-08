#include "bdd_collection/bdd_collection.h"
#include <queue>
#include <cassert>
#include <unordered_set>
#include <unordered_map>

namespace BDD {

    size_t bdd_collection::offset(const bdd_instruction& instr) const
    {
        assert(std::distance(&bdd_instructions[0], &instr) < bdd_instructions.size());
        assert(std::distance(&bdd_instructions[0], &instr) >= 0);
        return std::distance(&bdd_instructions[0], &instr); 
    }

    std::vector<size_t> bdd_collection::rebase_to_contiguous(const size_t bdd_nr)
    {
        const auto vars = variables(bdd_nr);
        std::unordered_map<size_t,size_t> var_map;
        var_map.reserve(vars.size());
        for(size_t i=0; i<vars.size(); ++i)
            var_map.insert({vars[i], i});
        rebase(bdd_nr, var_map);
        assert([&]() { std::vector<size_t> iota(vars.size()); std::iota(iota.begin(), iota.end(), 0); return variables(bdd_nr) == iota; }());
        return vars;
    }

    size_t bdd_collection::bdd_and(const size_t i, const size_t j)
    {
        return bdd_and(i, j, *this);
    }

    size_t bdd_collection::bdd_and(const size_t i, const size_t j, bdd_collection& o)
    {
        assert(i < nr_bdds());
        assert(j < nr_bdds());
        // TODO: allocate stack etc. locally
        assert(o.stack.empty());
        assert(o.generated_nodes.empty());
        assert(o.reduction.empty());

        // generate terminal vertices
        o.stack.push_back(bdd_instruction::botsink());
        o.stack.push_back(bdd_instruction::topsink());
        const size_t root_idx = bdd_and_impl(bdd_delimiters[i], bdd_delimiters[j], o);

        if(root_idx != std::numeric_limits<size_t>::max())
        {
            assert(o.stack.size() > 2);
            const size_t offset = o.bdd_delimiters.back();
            for(ptrdiff_t s = o.stack.size()-1; s>=0; --s)
            {
                const bdd_instruction bdd_stack = o.stack[s];
                const size_t lo = offset + o.stack.size() - bdd_stack.lo - 1;
                const size_t hi = offset + o.stack.size() - bdd_stack.hi - 1;
                o.bdd_instructions.push_back({lo, hi, o.stack[s].index});
            }
            o.bdd_delimiters.push_back(o.bdd_instructions.size());
            assert(o.is_bdd(o.bdd_delimiters.size()-2));
        }

        o.generated_nodes.clear();
        o.reduction.clear();
        o.stack.clear();
        if(root_idx == std::numeric_limits<size_t>::max())
            return std::numeric_limits<size_t>::max();
        return o.bdd_delimiters.size()-2;
    }

    size_t bdd_collection::bdd_and(const int i, const int j, bdd_collection& o)
    {
        assert(i >= 0 && j >= 0);
        return bdd_and(size_t(i), size_t(j), o);
    }
    size_t bdd_collection::bdd_and(const int i, const int j)
    {
        return bdd_and(size_t(i), size_t(j), *this); 
    }

    // given two bdd_instructions indices, compute new melded node, if it has not yet been created. Return index on stack.
    size_t bdd_collection::bdd_and_impl(const size_t f_i, const size_t g_i, bdd_collection& o)
    {
        // first, check whether node has been generated already
        if(o.generated_nodes.count({f_i,g_i}) > 0)
            return o.generated_nodes.find({f_i,g_i})->second;

        const bdd_instruction& f = bdd_instructions[f_i];
        const bdd_instruction& g = bdd_instructions[g_i];

        // check if instructions are terminals
        // TODO: more efficient
        if(f.is_terminal() && g.is_terminal())
        {
            if(f.is_topsink() && g.is_topsink())
                return 1; //topsink position on stack
            else
                return 0; //botsink position on stack
        }
        else if(f.is_terminal() && !g.is_terminal())
        {
            if(f.is_botsink())
                return 0; //botsink position on stack
            else if(!f.is_terminal() && g.is_terminal() && g.is_botsink()) // TODO: is superfluous!
                    return 0; //botsink position on stack
        }

        // compute lo and hi and see if they are present already. It not, add new branch instruction
        const size_t v = std::min(f.index, g.index);
        const size_t lo = bdd_and_impl(
                v == f.index ? f.lo : f_i,
                v == g.index ? g.lo : g_i,
                o
                );
        if(lo == std::numeric_limits<size_t>::max())
            return std::numeric_limits<size_t>::max();
        const size_t hi = bdd_and_impl(
                v == f.index ? f.hi : f_i,
                v == g.index ? g.hi : g_i,
                o
                );
        if(hi == std::numeric_limits<size_t>::max())
            return std::numeric_limits<size_t>::max();

        if(lo == hi)
            return lo;

        if(o.reduction.count({lo,hi,v}) > 0)
            return o.reduction.find({lo,hi,v})->second;

        o.stack.push_back({lo, hi, v});
        const size_t meld_idx = o.stack.size()-1;
        o.generated_nodes.insert(std::make_pair(std::array<size_t,2>{f_i,g_i}, meld_idx));
        o.reduction.insert(std::make_pair(bdd_instruction{lo,hi,v}, meld_idx));

        return meld_idx;
    }

    template<size_t N>
        size_t bdd_collection::bdd_and(const std::array<size_t,N>& bdds)
        {
            return bdd_and(bdds, *this);
        }

    template<size_t N>
        size_t bdd_collection::bdd_and(const std::array<size_t,N>& bdds, bdd_collection& o)
        {
            for(const size_t bdd_nr : bdds)
            {
                assert(bdd_nr < nr_bdds());
            }
            assert(o.stack.empty());
            assert(o.generated_nodes.empty());
            assert(o.reduction.empty());

            std::array<size_t,N> bdd_indices;
            for(size_t i=0; i<N; ++i)
                bdd_indices[i] = bdd_delimiters[bdds[i]];

            // generate terminal vertices
            o.stack.push_back(bdd_instruction::botsink());
            o.stack.push_back(bdd_instruction::topsink());
            std::unordered_map<std::array<size_t,N>,size_t,array_hasher<N>> generated_nodes;
            const size_t root_idx = bdd_and_impl(bdd_indices, generated_nodes, o);

            if(root_idx != std::numeric_limits<size_t>::max())
            {
                assert(o.stack.size() > 2);
                const size_t offset = o.bdd_delimiters.back();
                for(ptrdiff_t s = o.stack.size()-1; s>=0; --s)
                {
                    const bdd_instruction bdd_stack = o.stack[s];
                    const size_t lo = offset + o.stack.size() - bdd_stack.lo - 1;
                    const size_t hi = offset + o.stack.size() - bdd_stack.hi - 1;
                    o.bdd_instructions.push_back({lo, hi, o.stack[s].index});
                }
                o.bdd_delimiters.push_back(o.bdd_instructions.size());
                assert(o.is_bdd(o.bdd_delimiters.size()-2));
            }

            //generated_nodes.clear();
            o.reduction.clear();
            o.stack.clear();
            if(root_idx == std::numeric_limits<size_t>::max())
                return std::numeric_limits<size_t>::max();
            return o.bdd_delimiters.size()-2;
        }

    // explicit instantiation of bdd_and
    template size_t bdd_collection::bdd_and<3>(const std::array<size_t,3>& bdds, bdd_collection& o);
    template size_t bdd_collection::bdd_and<4>(const std::array<size_t,4>& bdds, bdd_collection& o);
    template size_t bdd_collection::bdd_and<5>(const std::array<size_t,5>& bdds, bdd_collection& o);
    template size_t bdd_collection::bdd_and<6>(const std::array<size_t,6>& bdds, bdd_collection& o);
    template size_t bdd_collection::bdd_and<7>(const std::array<size_t,7>& bdds, bdd_collection& o);
    template size_t bdd_collection::bdd_and<8>(const std::array<size_t,8>& bdds, bdd_collection& o);
    template size_t bdd_collection::bdd_and<9>(const std::array<size_t,9>& bdds, bdd_collection& o);

    template size_t bdd_collection::bdd_and<10>(const std::array<size_t,10>& bdds, bdd_collection& o);
    template size_t bdd_collection::bdd_and<11>(const std::array<size_t,11>& bdds, bdd_collection& o);
    template size_t bdd_collection::bdd_and<12>(const std::array<size_t,12>& bdds, bdd_collection& o);
    template size_t bdd_collection::bdd_and<13>(const std::array<size_t,13>& bdds, bdd_collection& o);
    template size_t bdd_collection::bdd_and<14>(const std::array<size_t,14>& bdds, bdd_collection& o);
    template size_t bdd_collection::bdd_and<15>(const std::array<size_t,15>& bdds, bdd_collection& o);
    template size_t bdd_collection::bdd_and<16>(const std::array<size_t,16>& bdds, bdd_collection& o);
    template size_t bdd_collection::bdd_and<17>(const std::array<size_t,17>& bdds, bdd_collection& o);
    template size_t bdd_collection::bdd_and<18>(const std::array<size_t,18>& bdds, bdd_collection& o);
    template size_t bdd_collection::bdd_and<19>(const std::array<size_t,19>& bdds, bdd_collection& o);

    template size_t bdd_collection::bdd_and<20>(const std::array<size_t,20>& bdds, bdd_collection& o);
    template size_t bdd_collection::bdd_and<21>(const std::array<size_t,21>& bdds, bdd_collection& o);
    template size_t bdd_collection::bdd_and<22>(const std::array<size_t,22>& bdds, bdd_collection& o);
    template size_t bdd_collection::bdd_and<23>(const std::array<size_t,23>& bdds, bdd_collection& o);
    template size_t bdd_collection::bdd_and<24>(const std::array<size_t,24>& bdds, bdd_collection& o);
    template size_t bdd_collection::bdd_and<25>(const std::array<size_t,25>& bdds, bdd_collection& o);
    template size_t bdd_collection::bdd_and<26>(const std::array<size_t,26>& bdds, bdd_collection& o);
    template size_t bdd_collection::bdd_and<27>(const std::array<size_t,27>& bdds, bdd_collection& o);
    template size_t bdd_collection::bdd_and<28>(const std::array<size_t,28>& bdds, bdd_collection& o);
    template size_t bdd_collection::bdd_and<29>(const std::array<size_t,29>& bdds, bdd_collection& o);

    template size_t bdd_collection::bdd_and<30>(const std::array<size_t,30>& bdds, bdd_collection& o);
    template size_t bdd_collection::bdd_and<31>(const std::array<size_t,31>& bdds, bdd_collection& o);
    template size_t bdd_collection::bdd_and<32>(const std::array<size_t,32>& bdds, bdd_collection& o);
    template size_t bdd_collection::bdd_and<33>(const std::array<size_t,33>& bdds, bdd_collection& o);
    template size_t bdd_collection::bdd_and<34>(const std::array<size_t,34>& bdds, bdd_collection& o);
    template size_t bdd_collection::bdd_and<35>(const std::array<size_t,35>& bdds, bdd_collection& o);
    template size_t bdd_collection::bdd_and<36>(const std::array<size_t,36>& bdds, bdd_collection& o);
    template size_t bdd_collection::bdd_and<37>(const std::array<size_t,37>& bdds, bdd_collection& o);
    template size_t bdd_collection::bdd_and<38>(const std::array<size_t,38>& bdds, bdd_collection& o);
    template size_t bdd_collection::bdd_and<39>(const std::array<size_t,39>& bdds, bdd_collection& o);

    template size_t bdd_collection::bdd_and<40>(const std::array<size_t,40>& bdds, bdd_collection& o);
    template size_t bdd_collection::bdd_and<41>(const std::array<size_t,41>& bdds, bdd_collection& o);
    template size_t bdd_collection::bdd_and<42>(const std::array<size_t,42>& bdds, bdd_collection& o);
    template size_t bdd_collection::bdd_and<43>(const std::array<size_t,43>& bdds, bdd_collection& o);
    template size_t bdd_collection::bdd_and<44>(const std::array<size_t,44>& bdds, bdd_collection& o);
    template size_t bdd_collection::bdd_and<45>(const std::array<size_t,45>& bdds, bdd_collection& o);
    template size_t bdd_collection::bdd_and<46>(const std::array<size_t,46>& bdds, bdd_collection& o);
    template size_t bdd_collection::bdd_and<47>(const std::array<size_t,47>& bdds, bdd_collection& o);
    template size_t bdd_collection::bdd_and<48>(const std::array<size_t,48>& bdds, bdd_collection& o);
    template size_t bdd_collection::bdd_and<49>(const std::array<size_t,49>& bdds, bdd_collection& o);

    // given two bdd_instructions indices, compute new melded node, if it has not yet been created. Return index on stack.
    template<size_t N>
    size_t bdd_collection::bdd_and_impl(const std::array<size_t,N>& bdds, std::unordered_map<std::array<size_t,N>,size_t,array_hasher<N>>& generated_nodes, bdd_collection& o)
    {
        // first, check whether node has been generated already
        if(generated_nodes.count(bdds) > 0)
            return generated_nodes.find(bdds)->second;

        std::array<bdd_instruction,N> bdd_instrs;
        for(size_t i=0; i<N; ++i)
            bdd_instrs[i] = bdd_instructions[bdds[i]];

        // check if all instructions are true
        const bool all_true_terminal = [&]() {
            for(bdd_instruction& f : bdd_instrs)
                if(!f.is_topsink())
                    return false;
            return true;
        }();
        if(all_true_terminal)
            return 1;

        const bool one_false_terminal = [&]() {
            for(bdd_instruction& f : bdd_instrs)
                if(f.is_botsink())
                    return true;
            return false; 
        }();
        if(one_false_terminal)
            return 0;

        const size_t v = [&]() {
            size_t idx = std::numeric_limits<size_t>::max();
            for(const bdd_instruction& f : bdd_instrs)
                idx = std::min(idx, f.index);
            return idx;
        }();

        std::array<size_t,N> lo_nodes;
        for(size_t i=0; i<N; ++i)
        {
            const bdd_instruction& f = bdd_instrs[i];
            lo_nodes[i] = f.index == v ? f.lo : bdds[i];
        }
        const size_t lo = bdd_and_impl(lo_nodes, generated_nodes, o);
        if(lo == std::numeric_limits<size_t>::max())
            return std::numeric_limits<size_t>::max();

        std::array<size_t,N> hi_nodes;
        for(size_t i=0; i<N; ++i)
        {
            const bdd_instruction& f = bdd_instrs[i];
            hi_nodes[i] = f.index == v ? f.hi : bdds[i];
        }
        const size_t hi = bdd_and_impl(hi_nodes, generated_nodes, o);
        if(hi == std::numeric_limits<size_t>::max())
            return std::numeric_limits<size_t>::max();

        if(lo == hi)
            return lo;

        if(o.reduction.count({lo,hi,v}) > 0)
            return o.reduction.find({lo,hi,v})->second;

        o.stack.push_back({lo, hi, v});
        const size_t meld_idx = o.stack.size()-1;
        generated_nodes.insert(std::make_pair(bdds, meld_idx));
        o.reduction.insert(std::make_pair(bdd_instruction{lo,hi,v}, meld_idx));

        return meld_idx;
    }

    size_t bdd_collection::add_bdd(node_ref root)
    {
        assert(bdd_delimiters.back() == bdd_instructions.size());

        auto nodes = root.nodes_postorder();
        std::reverse(nodes.begin(), nodes.end());
        for(size_t i=0; i<nodes.size(); ++i)
        {
            assert(!nodes[i].is_terminal());
            node_ref_hash.insert({nodes[i], bdd_instructions.size()});
            bdd_instructions.push_back(bdd_instruction{std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max(), nodes[i].variable()});
        }

        // TODO: not most efficient, record top and botsink above
        node_ref_hash.insert({nodes.back().botsink(), bdd_instructions.size()});
        bdd_instructions.push_back(bdd_instruction::botsink());
        assert(bdd_instructions.back().is_botsink());

        node_ref_hash.insert({nodes.back().topsink(), bdd_instructions.size()});
        bdd_instructions.push_back(bdd_instruction::topsink());
        assert(bdd_instructions.back().is_topsink());

        const size_t offset = bdd_delimiters.back();
        for(size_t i=0; i<nodes.size(); ++i)
        {
            assert(node_ref_hash.count(nodes[i].low()) > 0);
            assert(node_ref_hash.count(nodes[i].high()) > 0);
            assert(i < node_ref_hash.find(nodes[i].low())->second);
            assert(i < node_ref_hash.find(nodes[i].high())->second);
            bdd_instructions[offset + i].lo = node_ref_hash.find(nodes[i].low())->second;
            bdd_instructions[offset + i].hi = node_ref_hash.find(nodes[i].high())->second;
        }

        bdd_delimiters.push_back(bdd_instructions.size());

        // clean-up
        node_ref_hash.clear();
        assert(is_bdd(bdd_delimiters.size()-2));
        assert(nr_bdd_nodes(bdd_delimiters.size()-2) == nodes.size()+2);
        return bdd_delimiters.size()-2;
    }

    node_ref bdd_collection::export_bdd(bdd_mgr& mgr, const size_t bdd_nr) const
    {
        assert(bdd_nr < nr_bdds());
        assert(nr_bdd_nodes(bdd_nr) > 2);
        // TODO: use vector and shift indices by offset
        std::unordered_map<size_t, node_ref> bdd_instr_hash;
        assert(bdd_instructions[bdd_delimiters[bdd_nr+1]-2].is_terminal());
        assert(bdd_instructions[bdd_delimiters[bdd_nr+1]-1].is_terminal());
        for(ptrdiff_t i=bdd_delimiters[bdd_nr+1]-3; i>=ptrdiff_t(bdd_delimiters[bdd_nr]); --i)
        {
            const bdd_instruction& bdd_instr = bdd_instructions[i];
            assert(!bdd_instr.is_terminal());
            auto get_node_ref = [&](const size_t i) -> node_ref {
                assert(i < bdd_instructions.size());

                if(bdd_instructions[i].is_botsink())
                {
                    return mgr.botsink();
                }

                if(bdd_instructions[i].is_topsink())
                {
                    return mgr.topsink();
                }

                assert(i >= bdd_delimiters[bdd_nr]);
                assert(i < bdd_delimiters[bdd_nr+1]);
                assert(bdd_instr_hash.count(i) > 0);
                return bdd_instr_hash.find(i)->second;
            };

            node_ref lo = get_node_ref(bdd_instr.lo);
            node_ref hi = get_node_ref(bdd_instr.hi);
            node_ref bdd = mgr.unique_find(bdd_instr.index, lo, hi);
            assert(bdd_instr_hash.count(i) == 0);
            bdd_instr_hash.insert({i, bdd});
        }
        assert(bdd_instr_hash.count(bdd_delimiters[bdd_nr]) > 0);
        return bdd_instr_hash.find(bdd_delimiters[bdd_nr])->second;
    }

    size_t bdd_collection::add_bdd_impl(node_ref bdd)
    {
        if(node_ref_hash.count(bdd) > 0)
            return node_ref_hash.find(bdd)->second;

        const size_t bdd_idx = bdd_instructions.size();
        bdd_instructions.push_back(bdd_instruction{std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max(), bdd.variable()});
        node_ref_hash.insert({bdd, bdd_idx});
        if(!bdd.is_terminal())
        {
            const size_t lo_idx = add_bdd_impl(bdd.low());
            const size_t hi_idx = add_bdd_impl(bdd.high());
            assert(lo_idx != hi_idx); 
            bdd_instructions[bdd_idx].lo = lo_idx;
            bdd_instructions[bdd_idx].hi = hi_idx;
        }
        else if(bdd.is_botsink())
            bdd_instructions[bdd_idx] = bdd_instruction::botsink();
        else    
        {
            assert(bdd.is_topsink());
            bdd_instructions[bdd_idx] = bdd_instruction::topsink();
        }

        return bdd_idx;
    }

    size_t bdd_collection::nr_bdd_nodes(const size_t i) const
    {
        assert(i < nr_bdds());
        return bdd_delimiters[i+1] - bdd_delimiters[i];
    }

    size_t bdd_collection::nr_bdd_nodes(const size_t bdd_nr, const size_t variable) const
    {
        assert(bdd_nr < nr_bdds());
        size_t nr_occurrences = 0;
        for(size_t i=bdd_delimiters[bdd_nr]; i<bdd_delimiters[bdd_nr+1]; ++i)
            if(bdd_instructions[i].index == variable)
                ++nr_occurrences;
        return nr_occurrences;
    }

    bdd_instruction* bdd_collection::begin(const size_t bdd_nr)
    {
        assert(bdd_nr < nr_bdds());
        return &bdd_instructions[bdd_delimiters[bdd_nr]];
    }

    bdd_instruction* bdd_collection::end(const size_t bdd_nr)
    {
        assert(bdd_nr < nr_bdds());
        return &bdd_instructions[bdd_delimiters[bdd_nr+1]-2];
    } 

    const bdd_instruction* bdd_collection::cbegin(const size_t bdd_nr) const
    {
        assert(bdd_nr < nr_bdds());
        return &bdd_instructions[bdd_delimiters[bdd_nr]];
    }

    const bdd_instruction* bdd_collection::cend(const size_t bdd_nr) const
    {
        assert(bdd_nr < nr_bdds());
        return &bdd_instructions[bdd_delimiters[bdd_nr+1]-2];
    } 

    std::reverse_iterator<bdd_instruction*> bdd_collection::rbegin(const size_t bdd_nr)
    {
        assert(bdd_nr < nr_bdds());
        return std::reverse_iterator<bdd_instruction*>(&bdd_instructions[bdd_delimiters[bdd_nr+1]-2]); 
    }

    std::reverse_iterator<bdd_instruction*> bdd_collection::rend(const size_t bdd_nr)
    {
        assert(bdd_nr < nr_bdds());
        return std::reverse_iterator<bdd_instruction*>(&bdd_instructions[bdd_delimiters[bdd_nr]]);
    }

    bool bdd_collection::is_bdd(const size_t bdd_nr) const
    {
        return bdd_basic_check(bdd_nr) && no_parallel_arcs(bdd_nr) && has_no_isomorphic_subgraphs(bdd_nr);
    }

    bool bdd_collection::is_qbdd(const size_t bdd_nr) const
    {
        assert(bdd_basic_check(bdd_nr));
        assert(has_no_isomorphic_subgraphs(bdd_nr));
        return contiguous_vars(bdd_nr);
    }

    bool bdd_collection::bdd_basic_check(const size_t bdd_nr) const
    {
        assert(bdd_nr < nr_bdds());
        // check whether each bdd lo and hi pointers are directed properly and do not point to same node
        if(nr_bdd_nodes(bdd_nr) < 2) // otherwise terminals are not present
            return false;
        if(!bdd_instructions[bdd_delimiters[bdd_nr+1]-1].is_terminal())
            return false;
        if(!bdd_instructions[bdd_delimiters[bdd_nr+1]-2].is_terminal())
            return false;
        if(bdd_instructions[bdd_delimiters[bdd_nr+1]-2] == bdd_instructions[bdd_delimiters[bdd_nr+1]-1])
            return false;
        if(!(bdd_instructions[bdd_delimiters[bdd_nr+1]-2].is_botsink() || bdd_instructions[bdd_delimiters[bdd_nr+1]-1].is_botsink()))
            return false;
        if(!(bdd_instructions[bdd_delimiters[bdd_nr+1]-2].is_topsink() || bdd_instructions[bdd_delimiters[bdd_nr+1]-1].is_topsink()))
            return false;

        for(size_t i=bdd_delimiters[bdd_nr]; i<bdd_delimiters[bdd_nr+1]-2; ++i)
        {
            const bdd_instruction& bdd = bdd_instructions[i];
            if(bdd.is_terminal())
                return false;
            if(i >= bdd.lo || i >= bdd.hi)
                return false;
            if(bdd.lo >= bdd_delimiters[bdd_nr+1])
                return false;
            if(bdd.hi >= bdd_delimiters[bdd_nr+1])
                return false;
            if(bdd.lo < bdd_delimiters[bdd_nr])
                return false;
            if(bdd.hi < bdd_delimiters[bdd_nr])
                return false;
        }

        if(reachable_nodes(bdd_nr) != std::vector<char>(nr_bdd_nodes(bdd_nr),true))
            return false;
        return true; 
    }

    bool bdd_collection::no_parallel_arcs(const size_t bdd_nr) const
    {
        for(size_t i=bdd_delimiters[bdd_nr]; i<bdd_delimiters[bdd_nr+1]-2; ++i)
        {
            const bdd_instruction& bdd = bdd_instructions[i];
            if(bdd.lo == bdd.hi)
                return false;
        }

        return true;
    }

    bool bdd_collection::has_no_isomorphic_subgraphs(const size_t bdd_nr) const
    {
        std::unordered_set<bdd_instruction, bdd_instruction_hasher> bdd_nodes;
        for(size_t i=bdd_delimiters[bdd_nr]; i<bdd_delimiters[bdd_nr+1]-2; ++i)
        {
            const bdd_instruction& bdd = bdd_instructions[i];
            if(bdd_nodes.count(bdd) > 0)
                return false;
            bdd_nodes.insert(bdd);
        }

        return true; 
    }

    void bdd_collection::remove_parallel_arcs()
    {
        assert(nr_bdds() > 0);
        const size_t bdd_nr = nr_bdds() - 1;
        assert(bdd_basic_check(bdd_nr));
        if(no_parallel_arcs(bdd_nr))
            return;

        std::vector<char> remove(nr_bdd_nodes(bdd_nr), false);
        for(size_t idx=bdd_delimiters[bdd_nr]; idx<bdd_delimiters[bdd_nr+1]-2; ++idx)
        {
            const bdd_instruction& instr = bdd_instructions[idx];
            assert(instr.lo < bdd_instructions.size());
            assert(instr.hi < bdd_instructions.size());
            if(instr.lo == instr.hi)
                remove[idx - bdd_delimiters[bdd_nr]] = true;
        }

        for(std::ptrdiff_t idx=bdd_delimiters[bdd_nr+1]-3; idx>=bdd_delimiters[bdd_nr]; --idx)
        {
            if(!remove[idx - bdd_delimiters[bdd_nr]])
            {
                bdd_instruction& instr = bdd_instructions[idx];
                auto transitive_endpoint = [&](size_t i)
                {
                    while(remove[i - bdd_delimiters[bdd_nr]] == true)
                    {
                        assert(bdd_instructions[i].hi == bdd_instructions[i].lo);
                        i = bdd_instructions[i].lo;
                    }
                    return i;
                };

                instr.lo = transitive_endpoint(instr.lo);
                instr.hi = transitive_endpoint(instr.hi);
                if(instr.lo == instr.hi)
                    remove[idx - bdd_delimiters[bdd_nr]] = true;
                //assert(instr.lo != instr.hi);
            }
        }

        remove_dead_nodes(remove);
        assert(no_parallel_arcs(bdd_nr));
        reorder(bdd_nr);
        assert(no_parallel_arcs(bdd_nr));
    }

    void bdd_collection::reduce_isomorphic_subgraphs()
    {
        assert(nr_bdds() > 0);
        const size_t bdd_nr = nr_bdds() - 1;
        assert(is_bdd(bdd_nr) || is_qbdd(bdd_nr));

        std::vector<char> remove(nr_bdd_nodes(bdd_nr), false);
        std::unordered_map<bdd_instruction, size_t, bdd_instruction_hasher> bdd_map;
        bdd_map.insert({bdd_instructions[bdd_delimiters[bdd_nr+1]-1], bdd_delimiters[bdd_nr+1]-1});
        bdd_map.insert({bdd_instructions[bdd_delimiters[bdd_nr+1]-2], bdd_delimiters[bdd_nr+1]-2});

        for(std::ptrdiff_t idx=bdd_delimiters[bdd_nr+1]-3; idx>=bdd_delimiters[bdd_nr]; --idx)
        {
            bdd_instruction& instr = bdd_instructions[idx];
            auto it = bdd_map.find(instr);
            if(it == bdd_map.end())
                bdd_map.insert({instr, idx});
            else
                remove[idx - bdd_delimiters[bdd_nr]] = true;
            assert(bdd_map.count(bdd_instructions[instr.lo]) > 0);
            assert(bdd_map.count(bdd_instructions[instr.hi]) > 0);
            instr.lo = bdd_map.find(bdd_instructions[instr.lo])->second;
            instr.hi = bdd_map.find(bdd_instructions[instr.hi])->second;
        }

        remove_dead_nodes(remove);
        reorder(bdd_nr);
        assert(has_no_isomorphic_subgraphs(bdd_nr));
    }

    void bdd_collection::remove_dead_nodes(const std::vector<char>& remove)
    {
        assert(nr_bdds() > 0);
        const size_t bdd_nr = nr_bdds() - 1;
        assert(bdd_basic_check(bdd_nr));
        assert(remove.size() == nr_bdd_nodes(bdd_nr));
        assert(std::count(remove.begin(), remove.end(), false) > 0); // not all nodes are to be removed
        if(std::count(remove.begin(), remove.end(), true) == 0)
            return;

        // compactify: remove unused instructions and move remaining ones to contiguous placement
        std::vector<size_t> new_address;
        new_address.reserve(nr_bdd_nodes(bdd_nr)+2);
        size_t running_delta = 0;
        for(size_t i=0; i<nr_bdd_nodes(bdd_nr); ++i)
        {
            new_address.push_back(bdd_delimiters[bdd_nr] + i - running_delta);
            if(remove[i])
                ++running_delta;
        }

        // for sink nodes
        new_address.push_back(bdd_delimiters[bdd_nr] + new_address.size() - running_delta);
        new_address.push_back(bdd_delimiters[bdd_nr] + new_address.size() - running_delta);
            
        const size_t nr_remaining_instructions = std::count(remove.begin(), remove.end(), false);
        std::vector<bdd_instruction> new_bdd_instructions;
        new_bdd_instructions.reserve(nr_remaining_instructions);

        for(size_t idx=bdd_delimiters[bdd_nr]; idx<bdd_delimiters[bdd_nr+1]-2; ++idx)
        {
            if(remove[idx - bdd_delimiters[bdd_nr]])
                continue;
            bdd_instruction instr = bdd_instructions[idx];
            instr.lo = new_address[instr.lo-bdd_delimiters[bdd_nr]];
            instr.hi = new_address[instr.hi-bdd_delimiters[bdd_nr]]; 
            new_bdd_instructions.push_back(instr);
        }
        new_bdd_instructions.push_back(bdd_instructions[bdd_delimiters[bdd_nr+1]-2]);
        new_bdd_instructions.push_back(bdd_instructions[bdd_delimiters[bdd_nr+1]-1]);

        bdd_instructions.resize(bdd_delimiters[bdd_nr]);
        for(const auto instr : new_bdd_instructions)
            bdd_instructions.push_back(instr);
        bdd_delimiters.back() = bdd_instructions.size(); 

        assert(bdd_basic_check(bdd_nr));
    }

    std::vector<char> bdd_collection::reachable_nodes(const size_t bdd_nr) const
    {
        assert(bdd_nr < nr_bdds());
        std::vector<char> r(nr_bdd_nodes(bdd_nr), false);
        std::queue<size_t> q;
        q.push(0);
        while(!q.empty())
        {
            const size_t idx = q.front();
            q.pop();
            assert(idx < r.size());
            if(r[idx] == true)
                continue;
            r[idx] = true;
            const bdd_instruction& instr = bdd_instructions[bdd_delimiters[bdd_nr]+idx];
            if(!instr.is_terminal())
            {
                assert(instr.lo - bdd_delimiters[bdd_nr] < r.size());
                if(!r[instr.lo - bdd_delimiters[bdd_nr]])
                    q.push(instr.lo - bdd_delimiters[bdd_nr]);
                assert(instr.hi - bdd_delimiters[bdd_nr] < r.size());
                if(!r[instr.hi - bdd_delimiters[bdd_nr]])
                    q.push(instr.hi - bdd_delimiters[bdd_nr]); 
            }
        }

        return r;
    }

    void bdd_collection::reduce()
    {
        assert(nr_bdds() > 0);
        const size_t bdd_nr = nr_bdds() - 1;
        assert(bdd_basic_check(bdd_nr));

        remove_parallel_arcs();
        reduce_isomorphic_subgraphs();

        assert(is_bdd(bdd_nr));
    }

    bool bdd_collection::variables_sorted(const size_t bdd_nr) const
    {
        assert(bdd_nr < nr_bdds());

        for(size_t i=bdd_delimiters[bdd_nr]; i<bdd_delimiters[bdd_nr+1]-2; ++i)
        {
            const auto& bdd = bdd_instructions[i];
            assert(!bdd.is_terminal());
            const auto& lo = bdd_instructions[bdd.lo];
            if(!lo.is_terminal() && bdd.index > lo.index)
                return false;
            const auto& hi = bdd_instructions[bdd.hi];
            if(!hi.is_terminal() && bdd.index > hi.index)
                return false;
        }
        return true;
    }

    std::vector<size_t> bdd_collection::variables(const size_t bdd_nr) const
    {
        assert(bdd_nr < nr_bdds());

        bool are_vars_ordered = true;
        std::vector<size_t> vars;
        for(size_t i=bdd_delimiters[bdd_nr]; i<bdd_delimiters[bdd_nr+1]-2; ++i)
        {
            const auto& bdd = bdd_instructions[i];
            assert(!bdd.is_terminal());
            const auto& lo = bdd_instructions[bdd.lo];
            if(!lo.is_terminal() && bdd.index > lo.index)
                are_vars_ordered = false;
            const auto& hi = bdd_instructions[bdd.hi];
            if(!hi.is_terminal() && bdd.index > hi.index)
                are_vars_ordered = false;

            if(vars.size() > 0 && bdd_instructions[i].index == vars.back())
                continue;
            vars.push_back(bdd_instructions[i].index);
        }

        assert(vars.size() > 0);
        std::sort(vars.begin(), vars.end());
        vars.erase(std::unique(vars.begin(), vars.end()), vars.end());

        if(are_vars_ordered)
            return vars;

        std::vector<std::array<size_t,2>> arcs;
        arcs.reserve(2*nr_bdd_nodes(bdd_nr));
        for(size_t i=bdd_delimiters[bdd_nr]; i<bdd_delimiters[bdd_nr+1]-2; ++i)
        {
            const auto& bdd = bdd_instructions[i];
            assert(!bdd.is_terminal());
            const auto& lo = bdd_instructions[bdd.lo];
            if(!lo.is_terminal())
                arcs.push_back({bdd.index, lo.index});
            const auto& hi = bdd_instructions[bdd.hi];
            if(!hi.is_terminal())
                arcs.push_back({bdd.index, hi.index});
        }

        std::sort(arcs.begin(), arcs.end(), [](const auto a, const auto b) {
                if(a[0] == b[0])
                return a[1] < b[1];
                else
                return a[0] < b[0];
                });
        arcs.erase(std::unique(arcs.begin(), arcs.end()), arcs.end());

        std::unordered_map<size_t,size_t> adjacency_list_delimiters;
        adjacency_list_delimiters.reserve(vars.size());
        size_t last_var = std::numeric_limits<size_t>::max();
        for(size_t i=0; i<arcs.size(); ++i)
        {
            if(arcs[i][0] != last_var)
            {
                last_var = arcs[i][0];
                assert(adjacency_list_delimiters.count(arcs[i][0]) == 0);
                adjacency_list_delimiters.insert({arcs[i][0], i});
            }
        }
        assert(adjacency_list_delimiters.size() < vars.size()); // at least the last node is not recorded here (no outgoing nodes).

        // topological sort to get variable ordering
        std::queue<size_t> q;
        std::vector<size_t> vars_ordered;
        vars_ordered.reserve(vars.size());
        q.push(bdd_instructions[bdd_delimiters[bdd_nr]].index);
        // TODO: can be converted into vector
        std::unordered_map<size_t,size_t> incoming_counter;
        incoming_counter.reserve(vars.size());
        for(const auto a : arcs)
            incoming_counter[a[1]]++;
        assert(incoming_counter.size() == vars.size()-1); // otherwise some instruction is not reachable or there is more than one root node

        while(!q.empty())
        {
            const size_t var = q.front();
            q.pop();
            vars_ordered.push_back(var);
            auto adjacency_list_offset = adjacency_list_delimiters.find(var);
            if(adjacency_list_offset != adjacency_list_delimiters.end())
            {
                for(size_t i=adjacency_list_offset->second; i<arcs.size(); ++i)
                {
                    assert(i < arcs.size());
                    if(arcs[i][0] != var)
                        break;
                    const size_t next_var = arcs[i][1];
                    assert(incoming_counter.count(next_var) > 0);
                    assert(incoming_counter.find(next_var)->second > 0);
                    if(incoming_counter[next_var]-- == 1)
                        q.push(next_var);
                }
            }
        }
        
        assert(vars_ordered.size() == vars.size());
        return vars_ordered;
    }

    std::array<size_t,2> bdd_collection::min_max_variables(const size_t bdd_nr) const
    {
        assert(bdd_nr < nr_bdds());
        assert(nr_bdd_nodes(bdd_nr) > 0);
        size_t min_var = std::numeric_limits<size_t>::max();
        size_t max_var = 0;
        for(size_t i=bdd_delimiters[bdd_nr]; i<bdd_delimiters[bdd_nr+1]-2; ++i)
        {
            min_var = std::min(bdd_instructions[i].index, min_var);
            max_var = std::max(bdd_instructions[i].index, max_var); 
        }

        return {min_var, max_var}; 
    }

    size_t bdd_collection::root_variable(const size_t bdd_nr) const
    {
        assert(bdd_nr < nr_bdds());
        assert(nr_bdd_nodes(bdd_nr) > 0);
        assert(!bdd_instructions[bdd_delimiters[bdd_nr]].is_terminal());
        return bdd_instructions[bdd_delimiters[bdd_nr]].index;
    }

    void bdd_collection::remove(const size_t bdd_nr)
    {
        std::array<size_t,1> bdd_nrs{bdd_nr};
        remove(bdd_nrs.begin(), bdd_nrs.end());
    }

    std::array<std::vector<size_t>,2> bdd_collection::fixed_variables(const size_t bdd_nr) const
    {
        assert(bdd_nr < nr_bdds());
        std::vector<size_t> pos_fixed_vars;
        std::vector<size_t> neg_fixed_vars;

        assert(is_qbdd(bdd_nr));

        const auto vars = variables(bdd_nr);
        std::unordered_map<size_t,std::array<char,2>> fixed_vars_map;
        for(const size_t v : vars)
            fixed_vars_map.insert({v,{true,true}});

        for(size_t i=bdd_delimiters[bdd_nr]; i<bdd_delimiters[bdd_nr+1]-2; ++i)
        {
            const bdd_instruction& instr = bdd_instructions[i];
            if(!bdd_instructions[instr.lo].is_botsink())
                fixed_vars_map.find(instr.index)->second[0] = false;
            if(!bdd_instructions[instr.hi].is_botsink())
                fixed_vars_map.find(instr.index)->second[1] = false; 
        }

        for(const auto [var, fixations] : fixed_vars_map)
        {
            const auto [neg_fixed, pos_fixed] = fixations;
            if(neg_fixed)
                neg_fixed_vars.push_back(var);
            if(pos_fixed)
                pos_fixed_vars.push_back(var);
            assert(!(neg_fixed && pos_fixed)); 
        }

        return {neg_fixed_vars, pos_fixed_vars}; 
    }

    // Reorder nodes such that they are consecutive w.r.t. variables
    void bdd_collection::reorder(const size_t bdd_nr)
    {
        assert(bdd_nr < nr_bdds());

        if(is_reordered(bdd_nr))
            return;

        std::unordered_map<size_t,size_t> variable_counter;
        // TODO: can be made std::vector for efficiency
        std::unordered_map<size_t,size_t> node_permutation;

        std::vector<bdd_instruction> new_instructions(nr_bdd_nodes(bdd_nr));

        // copy terminals
        new_instructions[new_instructions.size()-2] = bdd_instructions[bdd_delimiters[bdd_nr+1]-2];
        new_instructions[new_instructions.size()-1] = bdd_instructions[bdd_delimiters[bdd_nr+1]-1];

        node_permutation.insert({bdd_delimiters[bdd_nr+1]-2, bdd_delimiters[bdd_nr+1]-2});
        node_permutation.insert({bdd_delimiters[bdd_nr+1]-1, bdd_delimiters[bdd_nr+1]-1});

        std::unordered_map<size_t,size_t> var_order;
        const auto vars = variables(bdd_nr);
        for(size_t i=0; i<vars.size(); ++i)
            var_order.insert({vars[i], i});

        // compute offsets
        for(size_t i=bdd_delimiters[bdd_nr]; i<bdd_delimiters[bdd_nr+1]-2; ++i)
            variable_counter[bdd_instructions[i].index]++;
        struct var_offset { size_t var; size_t offset; };
        std::vector<var_offset> var_offsets;
        var_offsets.reserve(variable_counter.size());
        for(const auto [var, nr_nodes] : variable_counter)
            var_offsets.push_back({var, nr_nodes});

        std::sort(var_offsets.begin(), var_offsets.end(), 
                [&](const auto& a, const auto b) {
                assert(var_order.count(a.var) > 0);
                assert(var_order.count(b.var) > 0);
                return var_order[a.var] < var_order[b.var];
                });
        {
            size_t prev_nr_vars = var_offsets[0].offset;
            var_offsets[0].offset = 0;
            for(size_t c=1; c<var_offsets.size(); ++c)
            {
                const size_t cur_nr_vars = var_offsets[c].offset;
                var_offsets[c].offset = var_offsets[c-1].offset + prev_nr_vars;
                prev_nr_vars = cur_nr_vars;
            }
        }

        for(const auto [var, offset] : var_offsets)
            variable_counter[var] = offset; 

        // fill new instructions, record new positions
        for(size_t i=bdd_delimiters[bdd_nr]; i<bdd_delimiters[bdd_nr+1]-2; ++i)
        {
            const size_t var = bdd_instructions[i].index;
            const size_t new_pos = variable_counter[var];
            assert(variable_counter.count(var) > 0);
            variable_counter.find(var)->second++;
            node_permutation.insert({i,new_pos + bdd_delimiters[bdd_nr]});
            new_instructions[new_pos] = bdd_instructions[i];
        }

        // reset offsets in instuctions
        for(size_t i=0; i<new_instructions.size()-2; ++i)
        {
            auto& instr = new_instructions[i];
            assert(node_permutation.count(instr.lo) > 0 && node_permutation.count(instr.hi) > 0);
            instr.lo = node_permutation[instr.lo];
            instr.hi = node_permutation[instr.hi]; 
        }

        // copy back
        std::copy(new_instructions.begin(), new_instructions.end(), bdd_instructions.begin() + bdd_delimiters[bdd_nr]); 

        //assert(is_bdd(bdd_nr)); // TODO: only test if it has no isomorphic sugraphs. It can be qbdd!
        assert(is_reordered(bdd_nr));
    }

    bool bdd_collection::is_reordered(const size_t bdd_nr) const
    {
        const auto vars = variables(bdd_nr);
        std::unordered_map<size_t,size_t> next_var_map;
        for(size_t i=0; i+1<vars.size(); ++i)
            next_var_map.insert({vars[i], vars[i+1]});
        next_var_map.insert({vars.back(), std::numeric_limits<size_t>::max()});

        for(size_t i=bdd_delimiters[bdd_nr]; i+1<bdd_delimiters[bdd_nr+1]-2; ++i)
        {
            const size_t var = bdd_instructions[i].index;
            const size_t next_var = bdd_instructions[i+1].index;
            if(var == next_var)
                continue;
            else if(next_var_map[var] == next_var)
                continue;
            else
                return false;
        }
        return true;
    }

    bdd_collection_entry bdd_collection::operator[](const size_t bdd_nr)
    {
        assert(bdd_nr < nr_bdds());
        return bdd_collection_entry(bdd_nr, *this);
    }

    bdd_instruction bdd_collection::operator()(const size_t bdd_nr, const size_t offset) const
    {
        assert(bdd_nr < nr_bdds());
        assert(offset < bdd_delimiters[bdd_nr+1]);
        assert(offset >= bdd_delimiters[bdd_nr]);
        return bdd_instructions[offset]; 
    }

    const bdd_instruction& bdd_collection::get_bdd_instruction(const size_t i) const
    {
        assert(i < bdd_instructions.size());
        return bdd_instructions[i];
    }

    size_t bdd_collection::new_bdd()
    {
        bdd_delimiters.push_back(bdd_instructions.size());
        return bdd_delimiters.size()-2;
    }

    bdd_collection_node bdd_collection::add_bdd_node(const size_t var)
    {
        bdd_instruction instr;
        instr.index = var;
        instr.lo = bdd_instruction::temp_undefined_index;
        instr.hi = bdd_instruction::temp_undefined_index;
        bdd_instructions.push_back(instr);
        bdd_delimiters.back()++;
        return bdd_collection_node(bdd_instructions.size()-1, bdd_collection_entry(bdd_delimiters.size()-2, *this));
    }

    void bdd_collection::close_bdd()
    {
        // add top and botsink
        bdd_instructions.push_back(bdd_instruction::botsink());
        bdd_instructions.push_back(bdd_instruction::topsink());

        // go over previous bdd instructions and reroute topsink and botsink entries to correct entries
        for(size_t idx=bdd_delimiters[bdd_delimiters.size()-2]; idx<bdd_instructions.size()-2; ++idx)
        {
            auto& instr = bdd_instructions[idx];

            if(instr.lo == bdd_instruction::temp_undefined_index)
                throw std::runtime_error("bdd lo arc not set");
            else if(instr.lo == bdd_instruction::temp_botsink_index)
                instr.lo = bdd_instructions.size()-2;
            else if(instr.lo == bdd_instruction::temp_topsink_index)
                instr.lo = bdd_instructions.size()-1;
            assert(instr.lo < bdd_instructions.size() && instr.lo >= bdd_delimiters[bdd_delimiters.size()-2]);

            if(instr.hi == bdd_instruction::temp_undefined_index)
                throw std::runtime_error("bdd hi arc not set");
            else if(instr.hi == bdd_instruction::temp_botsink_index)
                instr.hi = bdd_instructions.size()-2;
            else if(instr.hi == bdd_instruction::temp_topsink_index)
                instr.hi = bdd_instructions.size()-1;
            assert(instr.hi < bdd_instructions.size() && instr.hi >= bdd_delimiters[bdd_delimiters.size()-2]); 
        }

        bdd_delimiters.back() = bdd_instructions.size();
        reduce();
        assert(is_bdd(bdd_delimiters.size()-2));
    }

    bool bdd_collection::contiguous_vars(const size_t bdd_nr) const
    {
        assert(bdd_nr < nr_bdds());
        const auto vars = variables(bdd_nr);
        std::unordered_map<size_t,size_t> next_var_map;
        next_var_map.reserve(vars.size());
        for(size_t i=0; i+1<vars.size(); ++i)
            next_var_map.insert({vars[i], vars[i+1]});
        next_var_map.insert({vars.back(), bdd_instruction::topsink_index});

        for(size_t i=bdd_delimiters[bdd_nr]; i<bdd_delimiters[bdd_nr+1]-2; ++i)
        {
            const auto& bdd = bdd_instructions[i];
            assert(next_var_map.count(bdd.index) > 0);
            const size_t next_var = next_var_map.find(bdd.index)->second;

            const auto& low_bdd = bdd_instructions[bdd.lo];
            if(!(low_bdd.is_botsink() || low_bdd.index == next_var))
                return false;

            const auto& high_bdd = bdd_instructions[bdd.hi];
            if(!(high_bdd.is_botsink() || high_bdd.index == next_var))
                return false; 
        } 

        return true;
    }

    size_t bdd_collection::botsink_index(const size_t bdd_nr) const
    {
        assert(bdd_nr < nr_bdds());
        assert(
                (bdd_instructions[bdd_delimiters[bdd_nr+1]-1].is_topsink() && bdd_instructions[bdd_delimiters[bdd_nr+1]-2].is_botsink())
                ||
                (bdd_instructions[bdd_delimiters[bdd_nr+1]-2].is_topsink() && bdd_instructions[bdd_delimiters[bdd_nr+1]-1].is_botsink())
              ); 

        const size_t idx = bdd_instructions[bdd_delimiters[bdd_nr+1]-1].is_botsink() ? bdd_delimiters[bdd_nr+1]-1 : bdd_delimiters[bdd_nr+1]-2;
        assert(bdd_instructions[idx].is_botsink());
        return idx;
    }

    size_t bdd_collection::topsink_index(const size_t bdd_nr) const
    {
        assert(bdd_nr < nr_bdds());
        assert(
                (bdd_instructions[bdd_delimiters[bdd_nr+1]-1].is_topsink() && bdd_instructions[bdd_delimiters[bdd_nr+1]-2].is_botsink())
                ||
                (bdd_instructions[bdd_delimiters[bdd_nr+1]-2].is_topsink() && bdd_instructions[bdd_delimiters[bdd_nr+1]-1].is_botsink())
              ); 

        const size_t idx = bdd_instructions[bdd_delimiters[bdd_nr+1]-1].is_topsink() ? bdd_delimiters[bdd_nr+1]-1 : bdd_delimiters[bdd_nr+1]-2;
        assert(bdd_instructions[idx].is_topsink());
        return idx;
    }

    size_t bdd_collection::make_qbdd(const size_t bdd_nr, bdd_collection& o)
    {
        // rebase on contiguous
        const auto vars = rebase_to_contiguous(bdd_nr);

        assert(bdd_nr < nr_bdds());

        struct var_node_struct {
            size_t var;
            size_t bdd_node;
            bool operator==(const var_node_struct& o) const { return var == o.var && bdd_node == o.bdd_node; }
        };
        // TODO: use hash_helper
        struct var_node_hasher {
            size_t operator()(const var_node_struct& vn) const 
            { 
                const size_t var_hash = std::hash<size_t>()(vn.var);
                const size_t bdd_node_hash = std::hash<size_t>()(vn.bdd_node);
                return var_hash + 0x9e3779b9 + (bdd_node_hash << 6) + (bdd_node_hash >> 2);
            }
        };
        std::unordered_map<var_node_struct, size_t, var_node_hasher> bdd_index_map;

        const size_t botsink_idx = botsink_index(bdd_nr);
        const size_t topsink_idx = topsink_index(bdd_nr);

        bdd_index_map.insert({{std::numeric_limits<size_t>::max(), bdd_instruction::temp_topsink_index}, bdd_instruction::temp_topsink_index});
        std::vector<bdd_instruction> new_bdds;
        new_bdds.reserve(nr_bdd_nodes(bdd_nr));

        // add original bdds
        for(size_t i=bdd_delimiters[bdd_nr]; i<bdd_delimiters[bdd_nr+1]-2; ++i)
        {
            new_bdds.push_back(bdd_instructions[i]);
            if(new_bdds.back().lo == topsink_idx)
                new_bdds.back().lo = bdd_instruction::temp_topsink_index;
            else if(new_bdds.back().lo == botsink_idx)
                new_bdds.back().lo = bdd_instruction::temp_botsink_index;
            else
            {
                assert(new_bdds.back().lo >= bdd_delimiters[bdd_nr]);
                new_bdds.back().lo -= bdd_delimiters[bdd_nr];
            }

            if(new_bdds.back().hi == topsink_idx)
                new_bdds.back().hi = bdd_instruction::temp_topsink_index;
            else if(new_bdds.back().hi == botsink_idx)
                new_bdds.back().hi = bdd_instruction::temp_botsink_index;
            else
            {
                assert(new_bdds.back().hi >= bdd_delimiters[bdd_nr]);
                new_bdds.back().hi -= bdd_delimiters[bdd_nr];
            }

            bdd_index_map.insert({{bdd_instructions[i].index, i-bdd_delimiters[bdd_nr]}, new_bdds.size()-1});
        }

        // add intermediate bdds
        auto add_intermediate_nodes = [&](const size_t var, const size_t bdd_index) -> void
        {
            auto add_intermediate_nodes_impl = [&](const size_t var, const size_t bdd_index, auto& add_intermediate_nodes_ref) -> void
            {
                assert(bdd_index != bdd_instruction::temp_botsink_index);
                assert((var == std::numeric_limits<size_t>::max() && bdd_index == bdd_instruction::temp_topsink_index) || var < vars.size());
                if(bdd_index_map.count({var, bdd_index}) > 0)
                    return;
                new_bdds.push_back({});
                const size_t new_bdd_idx = new_bdds.size()-1;
                new_bdds[new_bdd_idx].index = var;
                const size_t next_var = var+1 == vars.size() ? std::numeric_limits<size_t>::max() : var+1;
                assert(var < vars.size());
                add_intermediate_nodes_ref(next_var, bdd_index, add_intermediate_nodes_ref);
                bdd_index_map.insert({{var, bdd_index}, new_bdd_idx});
                auto bdd_index_it = bdd_index_map.find({next_var, bdd_index});
                assert(bdd_index_it != bdd_index_map.end());
                new_bdds[new_bdd_idx].lo = bdd_index_it->second;
                new_bdds[new_bdd_idx].hi = bdd_index_it->second; 
            };
            add_intermediate_nodes_impl(var, bdd_index, add_intermediate_nodes_impl);
        };

        const size_t nr_bdd_nodes = new_bdds.size(); 

        for(size_t i=0; i<nr_bdd_nodes; ++i)
        {
            const size_t var = new_bdds[i].index;
            assert(var < vars.size());
            const size_t next_var = var+1 == vars.size() ? std::numeric_limits<size_t>::max() : var+1;
            if(new_bdds[i].lo != bdd_instruction::temp_botsink_index)
            {
                add_intermediate_nodes(next_var, new_bdds[i].lo);
                assert(bdd_index_map.count({next_var, new_bdds[i].lo}) > 0);
                new_bdds[i].lo = bdd_index_map.find({next_var, new_bdds[i].lo})->second;
            } 

            if(new_bdds[i].hi != bdd_instruction::temp_botsink_index)
            {
                add_intermediate_nodes(next_var, new_bdds[i].hi);
                assert(bdd_index_map.count({next_var, new_bdds[i].hi}) > 0);
                new_bdds[i].hi = bdd_index_map.find({next_var, new_bdds[i].hi})->second;
            } 
        }

        // add to end of nodes
        for(auto& bdd : new_bdds)
            o.bdd_instructions.push_back(bdd);

        // add terminal nodes
        o.bdd_instructions.push_back(bdd_instruction::topsink());
        o.bdd_instructions.push_back(bdd_instruction::botsink());
        o.bdd_delimiters.push_back(o.bdd_instructions.size());

        // update offsets
        for(size_t i=o.bdd_delimiters[o.bdd_delimiters.size()-2]; i<o.bdd_delimiters.back()-2; ++i)
        {
            auto& bdd = o.bdd_instructions[i];

            if(bdd.lo == bdd_instruction::temp_botsink_index)
                bdd.lo = o.bdd_delimiters.back()-1;
            else if(bdd.lo == bdd_instruction::temp_topsink_index)
                bdd.lo = o.bdd_delimiters.back()-2;
            else
                bdd.lo += o.bdd_delimiters[o.bdd_delimiters.size()-2];

            if(bdd.hi == bdd_instruction::temp_botsink_index)
                bdd.hi = o.bdd_delimiters.back()-1;
            else if(bdd.hi == bdd_instruction::temp_topsink_index)
                bdd.hi = o.bdd_delimiters.back()-2;
            else
                bdd.hi += o.bdd_delimiters[o.bdd_delimiters.size()-2];
        }

        const size_t new_bdd_nr = o.bdd_delimiters.size()-2;
        o.reorder(new_bdd_nr);

        // rebase back to original variables
        o.rebase(new_bdd_nr, vars.begin(), vars.end());
        rebase(bdd_nr, vars.begin(), vars.end());
        return new_bdd_nr;
    }

    size_t bdd_collection::make_qbdd(const size_t bdd_nr)
    {
        return make_qbdd(bdd_nr, *this);
    }

    void bdd_collection::append(const bdd_collection& o)
    {
        const size_t offset = bdd_instructions.size();
        for(size_t o_bdd_nr=0; o_bdd_nr<o.nr_bdds(); ++o_bdd_nr)
        {
            for(size_t j=o.bdd_delimiters[o_bdd_nr]; j<o.bdd_delimiters[o_bdd_nr+1]; ++j)
            {
                assert(j < o.bdd_instructions.size());
                bdd_instruction o_bdd = o.bdd_instructions[j];
                if(!o_bdd.is_terminal())
                {
                    o_bdd.lo += offset;
                    o_bdd.hi += offset;
                }
                bdd_instructions.push_back(o_bdd);
            }
            bdd_delimiters.push_back(bdd_instructions.size());
        }
    }

    //////////////////////////
    // bdd_collection_entry //
    //////////////////////////

    bdd_collection_entry::bdd_collection_entry(const size_t _bdd_nr, bdd_collection& _bdd_col)
        : bdd_nr(_bdd_nr),
        bdd_col(_bdd_col)
    {
        assert(bdd_nr < bdd_col.nr_bdds());
    }

    std::vector<size_t> bdd_collection_entry::variables()
    {
        return bdd_col.variables(bdd_nr);
    }

    size_t bdd_collection_entry::nr_nodes() const
    {
        return bdd_col.nr_bdd_nodes(bdd_nr);
    }

    size_t bdd_collection_entry::nr_nodes(const size_t variable) const
    {
        return bdd_col.nr_bdd_nodes(bdd_nr, variable);
    }

    bdd_collection_entry bdd_collection_entry::operator&(bdd_collection_entry& o)
    {
        assert(&bdd_col == &o.bdd_col);
        const size_t new_bdd_nr = bdd_col.bdd_and(bdd_nr, o.bdd_nr);
        return bdd_collection_entry(new_bdd_nr, bdd_col);
    }

    std::vector<size_t> bdd_collection_entry::rebase_to_contiguous()
    {
        return bdd_col.rebase_to_contiguous(bdd_nr);
    }

    bdd_collection_node bdd_collection_entry::root_node() const
    {
        return bdd_collection_node(bdd_col.bdd_delimiters[bdd_nr], *this);
    }

    bdd_collection_node bdd_collection_entry::first_node_postorder() const
    {
        assert(nr_nodes() > 2);
        return bdd_collection_node(bdd_col.bdd_delimiters[bdd_nr+1]-3, *this);
    }

    bdd_collection_node bdd_collection_entry::botsink() const
    {
        bdd_collection_node node(bdd_col.bdd_delimiters[bdd_nr+1]-2, *this);
        assert(node.is_botsink());
        return node;
    }

    bdd_collection_node bdd_collection_entry::topsink() const
    {
        bdd_collection_node node(bdd_col.bdd_delimiters[bdd_nr+1]-1, *this);
        assert(node.is_topsink());
        return node;
    }

    bdd_collection_node bdd_collection_entry::operator[](const size_t i) const
    {
        bdd_collection_node node( bdd_col.bdd_delimiters[bdd_nr] + i, *(this));
        return node;

    }

    /////////////////////////
    // bdd_collection_node //
    /////////////////////////

    bdd_collection_node::bdd_collection_node(const size_t _i, const bdd_collection_entry _bce)
        : i(_i),
        bce(_bce)
    {
        assert(i < bce.bdd_col.bdd_delimiters[bce.bdd_nr+1]);
        assert(i >= bce.bdd_col.bdd_delimiters[bce.bdd_nr]);
    }

    bdd_collection_node bdd_collection_node::lo() const
    {
        assert(!is_terminal());
        return {bce.bdd_col.bdd_instructions[i].lo, bce};
    }

    bdd_collection_node bdd_collection_node::hi() const
    {
        assert(!is_terminal());
        return {bce.bdd_col.bdd_instructions[i].hi, bce};
    }

    size_t bdd_collection_node::variable() const
    {
        return bce.bdd_col.bdd_instructions[i].index;
    }

    bool bdd_collection_node::is_botsink() const
    {
        return bce.bdd_col.bdd_instructions[i].is_botsink();
    }

    bool bdd_collection_node::is_topsink() const
    {
        return bce.bdd_col.bdd_instructions[i].is_topsink();
    }

    bool bdd_collection_node::is_terminal() const
    {
        assert(i < bce.bdd_col.bdd_instructions.size());
        return bce.bdd_col.bdd_instructions[i].is_terminal();
    }

    void bdd_collection_node::set_lo_arc(bdd_collection_node& node)
    {
        assert(i < bce.bdd_col.bdd_instructions.size());
        assert(i < node.i);
        assert(node.i < bce.bdd_col.bdd_instructions.size());
        auto& instr = bce.bdd_col.bdd_instructions[i];
        instr.lo = node.i;
    }

    void bdd_collection_node::set_hi_arc(bdd_collection_node& node)
    {
        assert(i < bce.bdd_col.bdd_instructions.size());
        assert(i < node.i);
        assert(node.i < bce.bdd_col.bdd_instructions.size());
        auto& instr = bce.bdd_col.bdd_instructions[i];
        instr.hi = node.i;
    }

    void bdd_collection_node::set_lo_to_0_terminal()
    {
        assert(i < bce.bdd_col.bdd_instructions.size());
        auto& instr = bce.bdd_col.bdd_instructions[i];
        instr.lo = bdd_instruction::temp_botsink_index;
    }

    void bdd_collection_node::set_lo_to_1_terminal()
    {
        assert(i < bce.bdd_col.bdd_instructions.size());
        auto& instr = bce.bdd_col.bdd_instructions[i];
        instr.lo = bdd_instruction::temp_topsink_index;
    }

    void bdd_collection_node::set_hi_to_0_terminal()
    {
        assert(i < bce.bdd_col.bdd_instructions.size());
        auto& instr = bce.bdd_col.bdd_instructions[i];
        instr.hi = bdd_instruction::temp_botsink_index;
    }

    void bdd_collection_node::set_hi_to_1_terminal()
    {
        assert(i < bce.bdd_col.bdd_instructions.size());
        auto& instr = bce.bdd_col.bdd_instructions[i];
        instr.hi = bdd_instruction::temp_topsink_index;
    }

    bdd_collection_node bdd_collection_node::next_postorder() const
    {
        assert(!is_terminal());
        assert(bce.root_node() != *this);
        return {i-1, bce};
    }

    bool bdd_collection_node::operator==(const bdd_collection_node& o) const
    {
        assert(&bce.bdd_col == &o.bce.bdd_col);
        return i == o.i;
    }

    bool bdd_collection_node::operator!=(const bdd_collection_node& o) const
    {
        return !(*this == o);
    }

    bdd_collection_node bdd_collection_node::operator=(const bdd_collection_node& o)
    {
        assert(&bce == &o.bce);
        i = o.i;
        return *this;
    }

    size_t bdd_collection::simplex_constraint(const size_t n)
    {
        assert(n > 0);

        if(n == 1)
        {
            bdd_instruction root;
            root.index = 0;
            root.lo = bdd_instructions.size()+1;
            root.hi = bdd_instructions.size()+2;
            bdd_instructions.push_back(root);

            bdd_instructions.push_back(bdd_instruction::botsink());
            bdd_instructions.push_back(bdd_instruction::topsink());

            bdd_delimiters.push_back(bdd_instructions.size());

            return nr_bdds()-1;
        }

        const size_t nr_bdd_nodes = 2*n-1;
        const size_t terminal_0_index = bdd_instructions.size() + nr_bdd_nodes;
        const size_t terminal_1_index = bdd_instructions.size() + nr_bdd_nodes + 1;
        const size_t offset = bdd_instructions.size();

        bdd_instruction root;
        root.index = 0;
        root.lo = offset+1;
        root.hi = offset+2;
        bdd_instructions.push_back(root);

        for(size_t i=1; i<n-1; ++i)
        {
            bdd_instruction instr_0;
            instr_0.index = i;
            instr_0.lo = offset + 2*i + 1;
            instr_0.hi = offset + 2*i + 2;
            bdd_instructions.push_back(instr_0);

            bdd_instruction instr_1;
            instr_1.index = i;
            instr_1.lo = offset + 2*i + 2;
            instr_1.hi = terminal_0_index;
            bdd_instructions.push_back(instr_1);
        }

        bdd_instruction instr_0;
        instr_0.index = n-1;
        instr_0.lo = terminal_0_index;
        instr_0.hi = terminal_1_index;
        bdd_instructions.push_back(instr_0);

        bdd_instruction instr_1;
        instr_1.index = n-1;
        instr_1.lo = terminal_1_index;
        instr_1.hi = terminal_0_index;
        bdd_instructions.push_back(instr_1);

        bdd_instructions.push_back(bdd_instruction::botsink());
        bdd_instructions.push_back(bdd_instruction::topsink());

        bdd_delimiters.push_back(bdd_instructions.size());

        return nr_bdds()-1;
    }

    size_t bdd_collection::not_all_false_constraint(const size_t n)
    {
        assert(n > 0);

        const size_t nr_bdd_nodes = n;
        const size_t terminal_0_index = bdd_instructions.size() + nr_bdd_nodes;
        const size_t terminal_1_index = bdd_instructions.size() + nr_bdd_nodes + 1;
        const size_t offset = bdd_instructions.size();

        for(size_t i=0; i<n-1; ++i)
        {
            bdd_instruction instr;
            instr.index = i;
            instr.lo = offset + i + 1;
            instr.hi = terminal_1_index;
            bdd_instructions.push_back(instr);
        }

        bdd_instruction instr;
        instr.index = n-1;
        instr.lo = terminal_0_index;
        instr.hi = terminal_1_index;
        bdd_instructions.push_back(instr);

        bdd_instructions.push_back(bdd_instruction::botsink());
        bdd_instructions.push_back(bdd_instruction::topsink());

        bdd_delimiters.push_back(bdd_instructions.size());
        return nr_bdds()-1;
    }

    size_t bdd_collection::all_equal_constraint(const size_t n)
    {
        // if n == 1 then this is just topsink
        assert(n > 1);

        const size_t nr_bdd_nodes = 2*n - 1;
        const size_t terminal_0_index = bdd_instructions.size() + nr_bdd_nodes;
        const size_t terminal_1_index = bdd_instructions.size() + nr_bdd_nodes + 1;
        const size_t offset = bdd_instructions.size();

        bdd_instruction first_instr;
        first_instr.index = 0;
        first_instr.lo = offset + 1;
        first_instr.hi = offset + 2;
        bdd_instructions.push_back(first_instr);

        for(size_t i=1; i+1<n; ++i)
        {
            bdd_instruction lo_instr;
            lo_instr.index = i;
            lo_instr.lo = offset + i*2 + 1;
            lo_instr.hi = terminal_0_index;
            bdd_instructions.push_back(lo_instr);

            bdd_instruction hi_instr;
            hi_instr.index = i;
            hi_instr.lo = terminal_0_index;
            hi_instr.hi = offset + i*2 + 2;
            bdd_instructions.push_back(hi_instr);
        }

        bdd_instruction last_lo_instr;
        last_lo_instr.index = n-1;
        last_lo_instr.lo = terminal_1_index;
        last_lo_instr.hi = terminal_0_index;
        bdd_instructions.push_back(last_lo_instr);

        bdd_instruction last_hi_instr;
        last_hi_instr.index = n-1;
        last_hi_instr.lo = terminal_0_index;
        last_hi_instr.hi = terminal_1_index;
        bdd_instructions.push_back(last_hi_instr);
        
        bdd_instructions.push_back(bdd_instruction::botsink());
        bdd_instructions.push_back(bdd_instruction::topsink());

        bdd_delimiters.push_back(bdd_instructions.size());
        assert(is_bdd(nr_bdds()-1));
        return nr_bdds()-1;
    }

}
