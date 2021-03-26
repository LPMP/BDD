#include "ILP_input.h"
#include "ILP_parser.h"
#include <fstream>

using namespace LPMP;

int main(int argc, char** argv)
{
    if(argc != 3)
        throw std::runtime_error("Two arguments expected: input file, output file");

    const ILP_input ilp = ILP_parser::parse_file(argv[1]);

    std::ofstream out_file;
    out_file.open (argv[2], std::ios::out | std::ios::trunc); 
    ilp.write_opb(out_file); 
}
