# -*- coding: utf-8 -*-
#
# *** Caution! ****
#  Without permission by the author, copy and redistribution of this program is PROHIBITED.  
#  下記の著者の許可を得ることなく、本プログラムをコピーまたは再配布することを禁じます。
# *****************
#
# Program: Data assimilation of phase-field simulation
# Application: Dendritic solidification in pure material
# Author: Prof. Akinori Yamanaka (Tokyo Univ. Agri. & Tech., Tokyo, JAPAN)
# Date: August, 2021

from ensemble_kalman_filter import EnsembleKalmanFilter # Ensemble Kalman filterのclass
import numpy as np
from numba import jit

# フェーズフィールドシミュレーションに関するパラメータ
xmax = 100 # x方向の差分格子点数
ymax = xmax # y方向の差分格子点数
n_grid = xmax * ymax # 全差分格子点数
dx = 2.0e-08 # 差分格子点の間隔[um]
dy = dx
dt = 5.0e-12 # 時間増分[s]
stepmax = 2000 # 時間ステップ数
stepobs = 100 # 観測時間間隔
stepout = stepobs # 結果出力間隔
pi = np.pi
delta = 4.0 * dx # 界面幅
gamma = 0.37 # 界面エネルギー(今回推定するパラメータ) [J/m2]
zeta = 0.03 # 異方性強度(今回推定するパラメータ)
aniso = 4.0 # 異方性モード数
angle0 = 0.0 # 優先成長方向
T_melt = 1728 # 融点[K]
K = 84.01 # 熱伝導率[W/(mK)]
c = 5.42e+06 # 比熱[J/K]
latent = 2.35e+09 # 潜熱 [J/mol]
lamb = 0.1
b = 2.0 * np.arctanh(1.0-2.0*lamb)
mu = 2.0 # 界面カイネティック係数 [m/(Ks)]
kappa = K / c # 熱拡散係数
eps = np.sqrt(3.0*delta*gamma/b) # 勾配エネルギー係数
W = 6.0 * gamma * b / delta # ダブルウェルポテンシャルのエネルギー障壁の高さ
pmobi = b * T_melt * mu / (3.0 * delta * latent) # フェーズフィールドモビリティー 
T_0 = 1424.5 # 温度[K]
init_size = 10 # 固相の初期核の大きさ[grid]

# データ同化に関するパラメータ
n_var = 2 # 状態変数の数。今回はフェーズフィールド（phi）と温度（temp）の2種類なので、n_var=2
var_shape = [(xmax, ymax), (xmax, ymax)] # 状態変数の配列の形 フェーズフィールドと温度の2つ
n_par = 2 # 推定するパラメータの数
dim_x = n_var*xmax*ymax + n_par # 状態ベクトルの長さ
dim_y = xmax*ymax # 観測ベクトルの長さ = 差分格子点と同じ位置・個数で観測データが得られたと仮定
Np = 20 # アンサンブルサイズ
par_init_m = np.array([0.045, 0.5]) # 推定するパラメータの初期平均値(異方性強度, 界面エネルギーの順)
par_init_sd = np.array([0.005, 0.05]) # 推定するパラメータの初期標準偏差
par_noise_sd = np.array([0.01*zeta, 0.01*gamma]) # システムノイズの標準偏差
sd_pert = 0.1 # 観測ノイズの標準偏差
rseed = 1000 # 乱数生成器のシード

# numpy配列の確保
phi = np.zeros((xmax,ymax)) # フェーズフィールド変数
temp = np.zeros((xmax,ymax)) # 温度変数
grad_phix = np.zeros((xmax,ymax)) 
grad_phiy = np.zeros((xmax,ymax))
eps2 = np.zeros((xmax,ymax))
lap_temp = np.zeros((xmax,ymax))
lap_phi = np.zeros((xmax,ymax))
ax = np.zeros((xmax,ymax))
ay = np.zeros((xmax,ymax))

# フェーズフィールドモデル計算 part 1 (勾配やラプラシアンの計算)
@jit(nopython=True)
def calcgrad(phi,temp,zeta,eps,W,grad_phix,grad_phiy,lap_phi,lap_temp,ax,ay,eps2):
    for j in range(ymax):
        for i in range(xmax):
            ip = i + 1
            im = i - 1
            jp = j + 1
            jm = j - 1
            if ip > xmax-1:
                ip = xmax - 1
            if im < 0:
                im = 0
            if jp > ymax-1:
                jp = ymax - 1
            if jm < 0:
                jm = 0

            grad_phix[i,j] = (phi[ip,j]-phi[im,j])/dx
            grad_phiy[i,j] = (phi[i,jp]-phi[i,jm])/dy
            lap_phi[i,j] = (2.*(phi[ip,j]+phi[im,j]+phi[i,jp]+phi[i,jm])+phi[ip,jp]+phi[im,jm]+phi[im,jp]+phi[ip,jm]-12.*phi[i,j])/(3.*dx*dx)
            lap_temp[i,j]= (2.*(temp[ip,j]+temp[im,j]+temp[i,jp]+temp[i,jm])+temp[ip,jp]+temp[im,jm]+temp[im,jp]+temp[ip,jm]-12.*temp[i,j])/(3.*dx*dx)

            if grad_phix[i,j] == 0.:
                if grad_phiy[i,j] > 0.:
                    angle = 0.5*pi
                else:
                    angle = -0.5*pi
            elif grad_phix[i,j] > 0.:
                if grad_phiy[i,j] > 0.:
                    angle = np.arctan(grad_phiy[i,j]/grad_phix[i,j])
                else:
                    angle = 2.0*pi + np.arctan(grad_phiy[i,j]/grad_phix[i,j])
            else:
                angle = pi + np.arctan(grad_phiy[i,j]/grad_phix[i,j])

            epsilon = eps*(1. + zeta * np.cos(aniso*(angle-angle0)))
            depsilon = -eps*aniso*zeta*np.sin(aniso*(angle-angle0))
            ay[i,j] = -epsilon * depsilon * grad_phiy[i,j]
            ax[i,j] =  epsilon * depsilon * grad_phix[i,j]
            eps2[i,j] = epsilon * epsilon

# フェーズフィールドモデル計算 part 2 (時間発展方程式の計算)
@jit(nopython=True)
def timeevol(phi,temp,zeta,eps,W,grad_phix,grad_phiy,lap_phi,lap_temp,ax,ay,eps2):
    for j in range(ymax):
        for i in range(xmax):
            ip = i + 1
            im = i - 1
            jp = j + 1
            jm = j - 1
            if ip > xmax-1:
                ip = xmax - 1
            if im < 0:
                im = 0
            if jp > ymax-1:
                jp = ymax -1
            if jm < 0:
                jm = 0

            dxdy = (ay[ip,j]-ay[im,j])/dx
            dydx = (ax[i,jp]-ax[i,jm])/dy
            grad_eps2x = (eps2[ip,j]-eps2[im,j])/dx
            grad_eps2y = (eps2[i,jp]-eps2[i,jm])/dy
            tet = phi[i,j]
            drive = -latent * (temp[i,j]-T_melt) / T_melt
            scal = grad_eps2x*grad_phix[i,j]+grad_eps2y*grad_phiy[i,j]

            chi = 0.0
            if tet > 0.0 and tet < 1.0:
                chi = np.random.uniform(-0.1,0.1)
            phi[i,j] = phi[i,j] + (dxdy + dydx + eps2[i,j]*lap_phi[i,j] + scal + 4.0*W*tet*(1.0-tet)*(tet-0.5+15.0/(2.0*W)*drive*tet*(1.0-tet)+chi))*dt*pmobi
            temp[i,j] = temp[i,j] + kappa*lap_temp[i,j]*dt + 30.0*tet*tet*(1.0-tet)*(1.0-tet)*(latent/c)*(phi[i,j]-tet)

# 計算結果をVTKファイルとして出力
def output(phi, temp, filename):
    f = open(filename,'w')
    f.write('# vtk DataFile Version 3.0 \n')
    f.write(filename+' \n')
    f.write('ASCII \n')
    f.write('DATASET STRUCTURED_POINTS \n')
    f.write('DIMENSIONS {0} {1} 1 \n'.format(xmax, ymax))
    f.write('ORIGIN 0.0 0.0 0.0 \n')
    f.write('SPACING 1.0 1.0 1.0 \n')
    f.write('POINT_DATA {} \n'.format(n_grid))
    f.write('SCALARS phasefield float \n')
    f.write('LOOKUP_TABLE default \n')
    [f.write('{:.10f} \n'.format(phi[l,m])) for l in range(xmax) for m in range(ymax)]
    f.write('SCALARS temperature float \n')
    f.write('LOOKUP_TABLE default \n')
    [f.write('{:.10f} \n'.format(temp[l,m])) for l in range(xmax) for m in range(ymax)]
    f.close()

@jit(nopython=True)
def fx(phi, temp, zeta, gamma): # fxの引数は状態変数の配列と推定するパラメータのみ(推定対象が変わる時は注意!)

    # 推定したパラメータでフェーズフィールドパラメータepsとWを再計算
    eps = np.sqrt(3.0*delta*gamma/b)
    W = 6.0 * gamma * b / delta

    # 時間発展の計算の途中で用いる配列は、fxの中で定義する(配列への代入を行わないなら外で定義してもよい)
    grad_phix = np.zeros((xmax,ymax))
    grad_phiy = np.zeros((xmax,ymax))
    eps2 = np.zeros((xmax,ymax))
    lap_temp = np.zeros((xmax,ymax))
    lap_phi = np.zeros((xmax,ymax))
    ax = np.zeros((xmax,ymax))
    ay = np.zeros((xmax,ymax))

    calcgrad(phi,temp,zeta,eps,W,grad_phix,grad_phiy,lap_phi,lap_temp,ax,ay,eps2)
    timeevol(phi,temp,zeta,eps,W,grad_phix,grad_phiy,lap_phi,lap_temp,ax,ay,eps2)

    return [phi, temp] # fxの戻り値は1 stepの計算後の状態変数の配列を含むリスト

# 観測ベクトルの定義(観測ベクトルの次元や種類が変わる時は注意!)
def hx(xt):
    return xt[0:dim_y] # 今回は観測データとしてPF変数を用いるので、hxは状態ベクトルからPF変数のみを取り出す

# ------ 以下メインプログラム --------

# フェーズフィールドと温度の初期値の設定
for j in range(0,ymax):
    for i in range(0,xmax):
        if i+j < init_size:
            phi[i,j] = 1.0
        temp[i,j] = T_0 + phi[i,j] * (T_melt-T_0)

# EnKFのclassの初期設定(詳細は、ensemble_data_assimilation.pyを参照)
EnKF = EnsembleKalmanFilter(dim_x=dim_x, dim_y=dim_y, Np=Np, fx=fx, hx=hx, n_var=n_var, var_shape=var_shape, n_par=n_par, \
    par_init_m=par_init_m, par_init_sd=par_init_sd, par_noise_sd=par_noise_sd, sd_pert=sd_pert, dt=dt, stepmax=stepmax, stepobs=stepobs, stepout=stepout, rseed=rseed)

EnKF.initialize_stvar(phi, temp) # 各アンサンブルメンバーのphiとtempに上で設定した初期値を代入する(全アンサンブルメンバーで同じ初期状態. パラメータは異なる値.)
EnKF.initialize_param() # 推定するパラメータの初期分布を作成する

nstep = 0
output(phi, temp, 'true{}.vtk'.format(nstep))
obsdata = []

# 双子実験の準備：パラメータを真値に設定した、擬似観測データを得るためのシミュレーションを行う. 
#               C言語などをコンパイルした実行ファイルをsubprocessを使って呼び出し、計算結果をobsdataにappendする方法でも良い. 
for nstep in range(1,stepmax+1):
    calcgrad(phi,temp,zeta,eps,W,grad_phix,grad_phiy,lap_phi,lap_temp,ax,ay,eps2)
    timeevol(phi,temp,zeta,eps,W,grad_phix,grad_phiy,lap_phi,lap_temp,ax,ay,eps2)
    if (nstep % stepobs) == 0:
        output(phi, temp, 'phi{}.vtk'.format(nstep))
        obsdata.append(phi.copy()) # 各時刻の擬似観測データ(PF変数)をリストに入れていく

EnKF.set_obsdata(obsdata) # 上記で計算した擬似観測データをオブジェクト内の変数に渡す
EnKF.run() # データ同化を実行(一期先予測とフィルタリングの繰り返し)

# バイナリで保存していた計算結果(numpy配列)をvtkファイル形式で出力
for i in range(0,stepmax+1,stepobs): 
    phin = np.load(file='varm0_{0}.npy'.format(i))
    tempn = np.load(file='varm1_{0}.npy'.format(i))
    output(phin, tempn, 'mean{}.vtk'.format(i)) # フェーズフィールド場と温度場のアンサンブル平均を出力

    phin = np.load(file='varsd0_{0}.npy'.format(i))
    tempn = np.load(file='varsd1_{0}.npy'.format(i))
    output(phin, tempn, 'sd{}.vtk'.format(i)) # フェーズフィールド場と温度場のアンサンブルの標準偏差を出力
