#pragma once

#include "ILP_parser.h"
#include "OPB_parser.h"
#include <regex>
#include "bdd_logging.h"

namespace LPMP {

    inline ILP_input parse_ilp_file(const std::string& filename)
    {
        // determine whether file is in LP format or in opb one.
        if(filename.substr(filename.find_last_of(".") + 1) == "opb")
        {
            bdd_log << "[bdd solver] Parse opb file\n";
            return OPB_parser::parse_file(filename);
        }
        else if(filename.substr(filename.find_last_of(".") + 1) == "lp")
        {
            bdd_log << "[bdd solver] Parse lp file\n";
            return ILP_parser::parse_file(filename);
        }
        else // peek into file
        {
            throw std::runtime_error("peeking into files not implemented yet, use either .lp or .opb file extension");
        }
    }

    inline ILP_input parse_ilp_string(const std::string& input)
    {
        // if file begins with * (i.e. opb comment) or with min: then it is an opb file
        std::regex comment_regex("^\\w*\\*");
        std::regex opb_min_regex("^\\w*min:");
        if(std::regex_search(input, comment_regex) || std::regex_search(input, opb_min_regex)) 
        {
            bdd_log << "[bdd solver] Parse opb string\n";
            return OPB_parser::parse_string(input); 
        }
        else
        {
            bdd_log << "[bdd solver] Parse lp string\n";
            return ILP_parser::parse_string(input); 
        }
    }
}