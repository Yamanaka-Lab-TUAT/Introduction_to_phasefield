# -*- coding: utf-8 -*-
#
# *** Caution! ****
#  Without permission by the author, copy and redistribution of this program is PROHIBITED.  
#  下記の著者の許可を得ることなく、本プログラムをコピーまたは再配布することを禁じます。
# *****************
#
# Program: Ensemble data assimilation of phase-field simulation
# Application: Dendritic solidification in pure material
# Author: Prof. Akinori Yamanaka (Tokyo Univ. Agri. & Tech., Tokyo, JAPAN)
# Date: August, 2021

import numpy as np
from numpy import array, zeros, ones, eye, dot
from numpy.random import normal

class EnsembleDA(object):
    """
    dim_x : Dimension of state vector(状態ベクトルの次元)
    dim_y : Dimension of observation vector(観測ベクトルの次元)
    Np : Ensemble size(アンサンブル数)
    fx : (状態ベクトルを更新するシミュレーションモデル. 状態空間モデルにおけるf(x))
        The function which updates state variables.
        Arguments must be numpy arrays for state variables(e.g. phase-field, 
        concentration, temperature) and parameters to estimate.
        Return value must be the list which includes all of numpy arrays for state variables.
    hx : Observation operator(観測演算子h_t)
    n_var : The number of kinds of state variables(状態変数の種類の個数)
    var_shape : The list which includes shapes (tuples) of state variables(状態ベクトルの大きさを含むリスト)
    n_par : The number of parameters to estimate(推定するパラメータの個数)
    par_init_m : (推定するパラメータの初期推定値のアンサンブルの平均値)
        Numpy array which includes initial mean for the parameters to estimate.
        Arraysize must be equal to n_par.
    par_init_sd : (推定するパラメータの初期推定値のアンサンブルの標準偏差)
        Numpy array which includes initial standard deviation for the parameters to estimate.
        Arraysize must be equal to n_par.
    par_noise_sd : (推定するパラメータに加えるシステムノイズv_tの標準偏差. 平均値は0.)
        Numpy array which includes standard deviation for system noise added to the parameters to estimate.
        Arraysize must be equal to n_par.
    sd_pert : Standard deviation for observation noise (観測ノイズw_tの標準偏差. 平均値は0.)
    dt : Time increment (シミュレーションの時間増分)
    stepmax : The number of all calculation steps (全時間ステップ数)
    stepobs : Interval of observation (観測データの時間間隔)
    stepout : Interval of output (計算結果の出力間隔)
    rseed : Seed value for random number generater (乱数発生のシード)
    """

    def __init__(self, dim_x, dim_y, Np, fx, hx, n_var, var_shape, n_par, par_init_m, par_init_sd, par_noise_sd, sd_pert, dt, stepmax, stepobs, stepout, rseed):
        """
        define numpy arrays. (使用するnumpy配列の宣言)
        initialize the arrays. (配列の初期設定やゼロ設定)
        """
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.Np = Np
        self.fx = fx
        self.hx = hx
        self.n_var = n_var
        self.var_shape = var_shape
        self.n_par = n_par
        self.par_init_m = par_init_m
        self.par_init_sd = par_init_sd
        self.par_noise_sd = par_noise_sd
        self.par_sv_idx = dim_x - n_par
        self.sd_pert = sd_pert
        self.dt = dt
        self.stepmax = stepmax
        self.stepobs = stepobs
        self.stepout = stepout
        self.Xt = zeros((dim_x, Np))
        self.Xtt = zeros((dim_x, Np))
        self.ytvec = zeros(dim_y)
        self.xt_mean = zeros(dim_x)
        self.xt_sd = zeros(dim_x)
        self.stvar = [[zeros(var_shape[i]) for i in range(n_var)] for j in range(Np)] # 状態ベクトル x_t
        self.var_size = [self.stvar[0][i].size for i in range(n_var)]
        self.i_start = zeros(n_var, dtype=int)
        self.i_end = zeros(n_var, dtype=int)
        self.i_end[0] = self.var_size[0]
        if n_var >= 2:
            for i in range(1, n_var):
                self.i_start[i] = self.i_end[i-1]
                self.i_end[i] = self.i_end[i-1] + self.var_size[i]

        np.random.seed(rseed)

    def initialize_stvar(self, *var):
        """
        initializes state variables. (状態ベクトルx_tのアンサンブルメンバーにフェーズフィールド変数と温度場の初期値を代入する)
        Initial values of state variables are the same value for all ensemble members. (本プログラムでは, 全てのアンサンブルで同じ初期状態とする)
        """

        for ip in range(self.Np):
            for i_var in range(self.n_var):
                self.stvar[ip][i_var][:] = var[i_var].copy()

    def initialize_param(self):
        """        
        initializes parameters to estimate. (推定対象となるパラメータの初期値のアンサンブルを拡大状態ベクトルX_tに代入する)
        """

        for i in range(self.n_par):
            e = normal(self.par_init_m[i], self.par_init_sd[i], self.Np)
            self.Xt[self.par_sv_idx+i, :] = e[:]

    def set_stvar_to_sv(self, i_var, ip):
        """
        sets state variables to state vector. (拡大状態ベクトルX_tにx_tの中身を代入する)
        """

        i_start = self.i_start[i_var]
        i_end = self.i_end[i_var]

        self.Xt[i_start:i_end, ip] = self.stvar[ip][i_var].flatten(order='F')

    def get_stvar_from_sv(self, i_var, ip):
        """
        gets state variables from state vector. (状態ベクトルから状態変数を抜き出す)
        """

        i_start = self.i_start[i_var]
        i_end = self.i_end[i_var]
        shape = self.var_shape[i_var]

        self.stvar[ip][i_var][:] = self.Xt[i_start:i_end, ip].reshape(shape, order='F').copy()

    def set_obsdata(self, obsdata):
        """
        obsdata must be the list which includes observation datas(numpy arrays). (観測ベクトルをnumpy配列に代入する)
        The number of observation datas must be equal to stepmax / stepobs. (観測データの数は、stepmax/stepobsの値と一致しなければならない)
        """

        self.obsdata = obsdata

    def set_obsvec(self, iobs):
        """ 
        sets observation data to observation vector. (観測データから抜き出したデータを観測ベクトルy_tに代入する. )
        """

        self.ytvec = self.obsdata[iobs-1].flatten(order='F')

    def calc_mean_and_sd(self):
        """ 
        calculates mean and standard deviation for state variables and parameters to estimate. (推定した状態変数とパラメータの平均値と標準偏差を計算する)
        """

        for i in range(self.dim_x):
            self.xt_mean[i] = np.mean(self.Xt[i, :])
            self.xt_sd[i] = np.std(self.Xt[i, :], ddof=1)

    def get_mean_and_sd(self, i_var):
        """
        gets mean and standard deviation for state variables. (推定した状態変数の平均値と標準偏差を計算する)
        """

        i_start = self.i_start[i_var]
        i_end = self.i_end[i_var]
        shape = self.var_shape[i_var]

        return self.xt_mean[i_start:i_end].reshape(shape, order='F').copy(), \
        self.xt_sd[i_start:i_end].reshape(shape, order='F').copy()

    def get_param_from_sv(self, ip):
        """ 
        gets parameters to estimate from state vector. (状態ベクトルから推定したパラメータを抜き出す)
        """     

        return self.Xt[self.par_sv_idx:self.dim_x, ip]

    def add_systemnoise(self):
        """ 
        adds system noise to parameters to estimate. (推定するパラメータにシステムノイズを加える)
        """

        Np = self.Np
        n_par = self.n_par
        par_noise_sd = self.par_noise_sd
        par_sv_idx = self.par_sv_idx

        for i in range(n_par):
            e = normal(0.0, par_noise_sd[i], Np)
            self.Xt[par_sv_idx+i, :] += e[:]

    def output(self, nstep):
        """ 
        outputs mean and standard deviation for state variables and parameters to estimate. (推定した状態変数やパラメータを出力する)
        """        

        Np = self.Np
        n_var = self.n_var
        n_par = self.n_par
        par_sv_idx = self.par_sv_idx
        dt = self.dt

        [self.set_stvar_to_sv(i_var, ip) for i_var in range(n_var) for ip in range(Np)]
        self.calc_mean_and_sd()

        for i in range(n_var):
            mean, sd = self.get_mean_and_sd(i)
            np.save(file='varm{0}_{1}.npy'.format(i, nstep), arr=mean) # 推定した状態変数の平均値(つまりアンサンブルの平均値)の出力
            np.save(file='varsd{0}_{1}.npy'.format(i, nstep), arr=sd) # 推定した状態変数の標準偏差(つまりアンサンブルの標準偏差)の出力

        for i in range(n_par):
            f = open('para{0}.csv'.format(i), 'a')
            if nstep == 0:
                f.write('time, para, +- \n')
            f.write('{0}, {1}, {2} \n'.format(nstep*dt, self.xt_mean[par_sv_idx+i], self.xt_sd[par_sv_idx+i])) # 推定したパラメータのアンサンブル平均値と標準偏差の出力
            f.close()

    def update(self):
        """ 
        updates state vector. (状態ベクトルの更新. つまりフェーズフィールドモデルを用いて一期先予測を行う)
        """  

        for i in range(self.Np):
            para = self.get_param_from_sv(i) # 状態ベクトルからパラメータを抜き出す
            self.stvar[i] = self.fx(*self.stvar[i][:], *para) # 抜き出したパラメータを使ってシミュレーションを実行(main.pyのfxを呼び出す)
                                                              # C言語などをコンパイルした実行ファイルをsubprocessを使って呼び出し、計算結果をstvarに代入する方法でも良い. 

        self.add_systemnoise() # シミュレーション結果に対してシステムノイズを加える

    def filtering(self):
        """
        calculates filtered PDF. This method is overriden in each sub class. (ensemble_kalman_filter.pyにあるfilteringを実行する)
        """

        pass

    def run(self):
        """
        implements data assimilation. (一期先予測とフィルタリングを繰り返すルーチン)
        """

        Np = self.Np
        n_var = self.n_var
        stepmax = self.stepmax
        stepobs = self.stepobs
        stepout = self.stepout

        self.output(0)

        for nstep in range(1, stepmax+1): # 一期先予測の時間ループ

            self.update() # 一期先予測の実行(つまりシミュレーションを1ステップ進める)

            if nstep % stepobs == 0: # 観測データが存在する時刻では, フィルタリングを実行する

                iobs = int(nstep / stepobs)
                [self.set_stvar_to_sv(i_var, ip) for i_var in range(n_var) for ip in range(Np)] # 状態ベクトルから状態変数を抜き出す
                self.set_obsvec(iobs) # 比較する観測ベクトルを設定
                self.filtering() # フィルタリングを実行し, 事後分布を求める
                [self.get_stvar_from_sv(i_var, ip) for i_var in range(n_var) for ip in range(Np)] # フィルタリングで修正された状態変数を状態ベクトルに戻す

            if nstep % stepout == 0: # 計算結果を出力する

                self.output(nstep)            
