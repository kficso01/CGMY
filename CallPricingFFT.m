 
function call_price_fft = CallPricingFFT(model,S,K,T,r,d,varargin)
% uses the file CharacteristicFunctionLib
%------------------------------------------
%--- FFT Option Pricing using Carr-Madan --
%--- Chapter 5                           --
%------------------------------------------
lnS = log(S);
lnK = log(K);

% compute the optimal alpha
%optAlpha = optimalAlpha(model,lnS,lnK,T,r,d,varargin{:});
optAlpha = 0.5;

DiscountFactor = exp(-r*T);

% predefined parameters
FFT_N = 2^18;                               % must be a power of two (2^14)
FFT_eta = 0.05;                             % spacing of psi integrand

% effective upper limit for integration
% uplim = FFT_N * FFT_eta;

FFT_lambda = (2 * pi) / (FFT_N * FFT_eta);  %spacing for log strike output
FFT_b = (FFT_N * FFT_lambda) / 2;           

uvec = 1:FFT_N;
%log strike levels ranging from lnS-b to lnS+b
ku = - FFT_b + FFT_lambda * (uvec - 1);    

jvec = 1:FFT_N;
vj = (jvec-1) * FFT_eta;


% Applying FFT
tmp = DiscountFactor * psi(model,vj,optAlpha,lnS,T,r,d,varargin{:}) ...
    .* exp(1i * vj * (FFT_b)) * FFT_eta;
% Apply the Simpson rule for integration
tmp = (tmp / 3) .* (3 + (-1).^jvec - ((jvec - 1) == 0) );
% Compute the Call price vector
cpvec = real(exp(-optAlpha .* ku) .* fft(tmp) / pi);        

indexOfStrike = floor((lnK + FFT_b)/FFT_lambda + 1); 
iset = max(indexOfStrike)+1:-1:min(indexOfStrike)-1;
xp = ku(iset);                                          % strikes
yp = cpvec(iset);                                       % prices
call_price_fft = real(interp1(xp,yp,lnK));              % output

end

%analytical formula 
function ret = psi(model,v,alpha,varargin)
  ret = exp(feval(@CharacteristicFunctionLib, model, ...
      v - (alpha + 1) * 1i,varargin{:})) ...
      ./ (alpha.^2 + alpha - v.^2 + 1i * (2 * alpha + 1) .* v);
end



