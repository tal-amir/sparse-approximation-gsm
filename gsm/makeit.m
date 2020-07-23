function out_filename = makeit(precision, use_fast_math)
source_filename = 'gsm_v5_1_mex.c';

% This function compiles the GSM .c source file to a matlab executable.
% It is written for MinGW v6.3 (Windows) and GCC v6.3.x (Linux). To use with
% other compilers, the compilation command may need to be tuned.

% Input arguments:
%
% precision: 'single', 'double', 'quad'. Default: double
%   Determines the numerical type used for intermediate calculations.
%   'single' - Use 32-bit single-precision floating point. In this case, input and output
%              to the mex function is also of class 'single'.
%   'double' - Use 64-bit double-precision floating point. Recommended.
%   'quad'   - All calculations are performed with quad-precision 128bit floating
%              point, using the C <quadmath.h> library. This yields very accurate results
%              (probably to within 1 ulp in double precision) but may incur up to 40-fold
%              slowdown. Normally this is only used for accuracy evaluation.
%
% use_fast_math: logical (true, false). Default: true
%   If set to true, the '-Ofast' optimization flag is used, which yields a moderate speedup.
%   If false, the more conservative '-O2' flag is used, and intermediate results are forced
%   to be stored in the memory rather than in CPU registers (which may be 80-bit rather
%   than 64-bit). This makes the arithmetic operations compatible with the IEEE 754 standard
%   and thus more consistent across different computers.
%   Note that setting this flag to false does not in itself guarantee cross-system 
%   consistency, as slight differences in the result may be caused by to different
%   implementations of <math.h> functions such as exp, log etc.
%   Note: This flag is irrelevant if precision = 'quad'
 
if ~exist('precision','var') || isempty(precision)
    precision = 'double';
end

if ~exist('use_fast_math','var') || isempty(use_fast_math)
    use_fast_math = false;
end

fprintf('Precision: %s\nFast math: %s\n', precision, string(use_fast_math));
fprintf('Compiling source file ''%s''...\n', source_filename);

%% Construct mex arguments
args = cell(0,1);

args = [args; {'-DNDEBUG'}];

switch precision
    case 'quad'
        args = [args; {'-Dgsm_intermediate_numerical_type=4'}];
        args = [args; {'LINKLIBS="$LINKLIBS -lquadmath"'}];
    case 'double'
        args = [args; {'-Dgsm_intermediate_numerical_type=2'}];
    case 'single'
        args = [args; {'-Dgsm_intermediate_numerical_type=1'}];
        args = [args; {'-Dgsm_single_precision_matlab_args=true'}];
    otherwise
        error('Invalid value passed in argument <precision> (can be ''quad'', ''double'' or ''single'')');
end

if strcmp(precision, 'quad')
    args = [args; {'COPTIMFLAGS="-O2 -fwrapv"'}];
elseif ~use_fast_math
    args = [args; {'COPTIMFLAGS="-O2 -ffloat-store -fwrapv"'}];
else
    args = [args; {'COPTIMFLAGS="-Ofast -mfpmath=sse -fwrapv"'}];
end


args = [args; {'CDEBUGFLAGS=""'}];
args = [args; {'LDDEBUGFLAGS=""'}];
args = [args; {'DEBUGFLAGS=""'}];
args = [args; {'LINKDEBUGFLAGS=""'}];
args = [args; {'CFLAGS="-fPIC -pipe -std=c99 -pedantic -Wall -Wextra -Wconversion -Wunsuffixed-float-constants"'}];

args = [args; {source_filename}];

%% Construct mex command
mexcmd = sprintf('mex_result = mex(args{1}');

for i=2:numel(args)
    mexcmd = sprintf('%s, args{%d}', mexcmd, i);
end

mexcmd = [mexcmd, ');'];

% Execute mex command
eval(mexcmd);

% Return output
[~,name,~] = fileparts(source_filename);
out_filename = [name, '.', mexext];

fprintf('Created file ''%s''\n', out_filename);

end
