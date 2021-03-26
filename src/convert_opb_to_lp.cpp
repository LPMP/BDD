#include "ILP_input.h"
#include "OPB_parser.h"
#include <fstream>

using namespace LPMP;

int main(int argc, char** argv)
{
    if(argc != 3)
        throw std::runtime_error("Two arguments expected: input file, output file");

    const ILP_input ilp = OPB_parser::parse_file(argv[1]);

    std::ofstream out_file;
    out_file.open (argv[2], std::ios::out | std::ios::trunc); 
    ilp.write_lp(out_file); 
}
