import pandas as pd
import scipy as sp
from scipy import optimize

r = 0.05
alpha = 1.1
Sigma = 0.25

def cf_log_cgmy(u, lnS, T, mu ,half_etasq, C, G, M, Y):
    
    omega = -C*gamma(-Y)*(np.power(M-1,Y)-np.power(M,Y)+np.power(G+1,Y)-np.power(G,Y ))
    print ("Omega is : %s"  % omega)
#    raw_input("Omega")
    phi_CGMY = C*T*gamma(-Y)*(np.power(M-1j*u,Y)-np.power(M,Y)+np.power(G+1j*u,Y)- np.power(G,Y))
    print ("phi_CGMY is : %s" % phi_CGMY) 
#    raw_input("Phi_CGMY")
    phi = 1j*u*(lnS + (mu+omega-half_etasq)*T) + phi_CGMY - half_etasq*np.power(u,2)
    print ("phi is: %s" % phi)
  #  raw_input("Phi")
    return np.exp(scale*phi)


def Call_price_CM_precalc(T,r,alpha,N=15):
    
    
    DiscountFactor = np.exp(-r*T)
    
    FFT_N = int(np.power(2,N))
    #print(FFT_N)
    FFT_eta = .05
    a
    FFT_lambda = (2*np.pi)/(FFT_N*FFT_eta)
    #print(FFT_lambda)
    
    FFT_b = (FFT_N*FFT_lambda)/2
    #print(FFT_b)
    
    uvec = np.linspace(1,FFT_N,FFT_N)
    #print(uvec[2])
    
    ku = -FFT_b + FFT_lambda * (uvec - 1)
    #print(ku)
    jvec = uvec
    
    vj = (uvec-1) * FFT_eta
    

    
    global GLOBALT
    global GLOBALVJ
    global GLOBALALPHA
    global GLOBALFFTB
    global GLOALFFTLAMBDA
    global GLOBALJVEC
    global GLOBALKU
    global GLOBALFAKTOR

    GLOBALT = T
    GLOBALVJ = vj
    GLOBALALPHA = alpha
    GLOBALFFTB = FFT_b
    GLOALFFTLAMBDA = FFT_lambda
    GLOBALJVEC = jvec
    GLOBALKU = ku
    GLOBALFAKTOR = DiscountFactor * np.exp(1j * vj * FFT_b) * FFT_eta
    
def Call_price_CM_CF(CF, lnS):    
    global GLOBALCPVEC
    tmp = GLOBALFAKTOR * psi(CF, GLOBALVJ, GLOBALALPHA, lnS, GLOBALT)
    tmp = (tmp/3) * (3 + np.power(-1, GLOBALJVEC) - ((GLOBALJVEC-1)==0))
    GLOBALCPVEC = np.real(np.exp(-GLOBALALPHA * GLOBALKU) * fftpack.fft(tmp) / np.pi )

    
def Call_price_CF_K(lnK):
    indexOfStrike = int(np.floor((lnK + GLOBALFFTB ) / GLOALFFTLAMBDA + 1))
    xp = [GLOBALKU[indexOfStrike],GLOBALKU[indexOfStrike + 1 ]]
    yp = [GLOBALCPVEC[indexOfStrike],GLOBALCPVEC[indexOfStrike + 1 ]]
    return np.interp(lnK, xp, yp)


def psi(CF, GLOBALVJ, GLOBALALPHA, lnS, GLOBALT):
    
    u=GLOBALVJ-(GLOBALALPHA*1j+1j)
    
    denom = GLOBALALPHA**2 + GLOBALALPHA - Sigma**2 + GLOBALVJ * 2 * GLOBALALPHA * 1j + 1j * GLOBALVJ

    return CF(u)/denom


def do_CM_CGMY_fit(odate):
    # Use CF of CGMY in the CM model: cf_log_cgmy(u, lnS, T, mu ,half_etasq, C, G, M, Y)
    
    Tmt = 1/12
    
    file2 = 'Futures.xlsx'
    xl2 = pd.ExcelFile(file2)
    stock = xl2.parse('Futures')
    stock = stock.set_index('date')
    
    
    S = stock.loc[odate] # Retrieve the current stock price on the option date
    Call_price_CM_precalc(T=Tmt,r=r,alpha=alpha,N=17)
    lnS = np.log(S)
    
    def minimize(param):
        
        
        CF=lambda u, lnS, T:\
        cf_log_cgmy(u=u, lnS=lnS, T=Tmt, mu=r, half_etasq=param[4], C=param[0], G=param[1], M=param[2], Y=param[3])
        
        
        
        Call_price_CM_CF(CF, lnS)
        
        sum = 0
        for i in range(0,len(prices_oom)):
            a = Call_price_CF_K(np.log(strike_oom[i])) - prices_oom[i]
            sum = sum+a*a
        return sum
    
    #bB = (round(S/100)-2)*100
    #bA = bB-300
    #bC = bB+400
    #bD = bC+300
    
    #Kprice = get_price_key(omatrix)
    # Fit on out of the money
    #call_oom = getcall(omatrix).query("@bA <= strike <= @bB | @bC <= strike <= @bD")
    #strike_oom = call_oom[’strike’].values
    #prices_oom = call_oom[Kprice].values
    
    file = 'Options1.xlsx'

    xl = pd.ExcelFile(file)

    #print(xl.sheet_names)

    options = xl.parse('Options1')

    #print(stock).head()

    prices_oom = options['Oprice']
    
    strike_oom = options['Strike']
    #print(a)

    
    
    bounds = ((1e-3,np.inf),(1e-3,np.inf),(1e-3,np.inf),(-np.inf,2-1e-3),(1e-3,1-1e-3))
    param = [24.79, 94.45, 95.79, 0.2495, 0]
    
    local_opt = sp.optimize.minimize(minimize, param, bounds=bounds)
    global_opt = sp.optimize.basinhopping(minimize, param, minimizer_kwargs={'bounds': bounds})
    
    return {'opt':global_opt, 'local_opt':local_opt}
    

do_CM_CGMY_fit('2018-08-08')
