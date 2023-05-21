# -*- coding: utf-8 -*-
#
# *** Caution! ****
#  Without permission by the author, copy and redistribution of this program is PROHIBITED.  
#  下記の著者の許可を得ることなく、本プログラムをコピーまたは再配布することを禁じます。
# *****************
#
# Program: Main Program of Ensemble Kalman Filter
# Application: Dendritic solidification in pure material
# Author: Prof. Akinori Yamanaka (Tokyo Univ. Agri. & Tech., Tokyo, JAPAN)
# Date: August, 2021

from ensemble_data_assimilation import EnsembleDA
import numpy as np
from numpy import array, zeros, ones, eye, dot
from numpy.random import normal


class EnsembleKalmanFilter(EnsembleDA):

    def __init__(self, dim_x, dim_y, Np, fx, hx, n_var, var_shape, n_par, par_init_m, par_init_sd, par_noise_sd, sd_pert, dt, stepmax, stepobs, stepout, rseed):

        super().__init__(dim_x, dim_y, Np, fx, hx, n_var, var_shape, n_par, par_init_m, par_init_sd, par_noise_sd, sd_pert, dt, stepmax, stepobs, stepout, rseed)

        self.Yt = zeros((dim_y, Np)) # 観測ベクトルのアンサンブル(Y_t)
        self.Wt = zeros((dim_y, Np)) # 観測ノイズのアンサンブル(w_t^(i))
        self.Wtt = zeros((dim_y, Np)) # 観測ベクトルに加える擾乱(W_t)
        # self.Rtinv = zeros((Np, dim_y))
        self.S = zeros((dim_y, Np)) # カルマンゲインを計算するための配列(補足資料参照)
        self.Mat1 = ones((Np, Np)) # 単位行列I
        self.Mpp1 = zeros((Np, Np)) # カルマンゲインを計算するための配列(補足資料参照)
        self.Mpp2 = zeros((Np, Np)) # カルマンゲインを計算するための配列(補足資料参照)
        self.Mpp3 = zeros((Np, Np)) # カルマンゲインを計算するための配列(補足資料参照)
        self.Mpy = zeros((Np, dim_y)) # カルマンゲインを計算するための配列(補足資料参照)
        self.inv = np.linalg.inv # 逆行列の計算

    def filtering(self):
        """ 
        calculates filtered PDF (EnKFに基づき事後分布を求める. つまりfilteringを実施する)
        """

        Np = self.Np # アンサンブルサイズ
        sd_pert = self.sd_pert # 観測ノイズの標準偏差(main.pyの56行目で設定)
        dim_y = self.dim_y # 観測ベクトルの次元
        hx = self.hx # 観測行列H_t

        for i in range(Np):
            self.Yt[:, i] = self.ytvec[:] # 同化する観測ベクトル(Y_t = [y_t, y_t, ..., y_t])を設定 

        for i in range(dim_y):
            self.Wt[i, :] = normal(0.0, sd_pert, Np) # 観測ノイズのアンサンブル(w_t^(i))の設定(平均0, 標準偏差はmain.pyの56行目で設定) 
            # self.Rtinv[:, i] = 1.0 / np.var(self.Wt[i, :], ddof=1)

        self.Wtt = self.Wt - (1.0/Np) * dot(self.Wt, self.Mat1) # 観測ベクトルに加える擾乱(W_t)を計算 

        self.Yt += self.Wtt # 観測ベクトルに擾乱を加える (Y_t + W_tを計算)
        # self.Yt[0:dim_y, 0:Np] -= self.Xt[0:dim_y, 0:Np]

        self.Yt[0:dim_y, 0:Np] -= array([hx(self.Xt[:, i]) for i in range(Np)]).T # イノベーション(Y_t + W_t - H_t*X_t|t-1)を計算

        self.Xtt = self.Xt - (1.0/Np) * dot(self.Xt, self.Mat1) # 各アンサンブルとアンサンブル平均の差(tilda(X)_t|t-1 = [x_t|t-1^(i) - bar(x)_t|t-1, ...])を計算
        # self.S[0:dim_y, 0:Np] = self.Xtt[0:dim_y, 0:Np]

        self.S[0:dim_y, 0:Np] = array([hx(self.Xtt[:, i]) for i in range(Np)]).T # S_t = (H_t * X_t|t-1)/sqrt(Np-1) (ただし1/sqrt(Np-1)で除するのは, 次の行でまとめて計算) 

        self.Mpy = self.S.T * (1.0/(sd_pert*sd_pert*(Np-1))) # S_t^T * R_t^(-1)  (ここで, R_tは対角行列とし, その成分はsd_part^2であることに注意. 1/(Np-1)となるのは、Z_t|t-1との積を考慮)
        # self.Mpy = (self.S.T * self.Rtinv) * (1.0/(Np-1))

        self.Mpp1 = eye(Np) + dot(self.Mpy, self.S) # I + S_t^T * R_t^(-1) * S_t

        self.Mpp2 = dot(self.Mpy, self.Yt) # S_t^T * R_t^(-1) * (Y_t + W_t - H_t*X_t|t-1)

        self.Mpp3 = dot(self.inv(self.Mpp1), self.Mpp2) # (I + S_t^T * R_t^(-1) * S_t)^-1 * S_t^T * R_t^(-1) * (Y_t + W_t - H_t*X_t|t-1)

        self.Xt += dot(self.Xtt, self.Mpp3) # アンサンブルメンバーを更新(X_t|t = X_t|t-1 + Z_t|t-1 * (I + S_t^T * R_t^(-1) * S_t)^-1 * S_t^T * R_t^(-1) * (Y_t + W_t - H_t*X_t|t-1)) を計算


