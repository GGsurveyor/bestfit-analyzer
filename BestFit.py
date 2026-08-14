import numpy as np
import pandas as pd

class BestFitEngine:
    @staticmethod
    def best_fit(measured, design):
        # 1. 计算质心
        centroid_m = np.mean(measured, axis=0)
        centroid_d = np.mean(design, axis=0)
        
        # 2. 中心化数据
        m_centered = measured - centroid_m
        d_centered = design - centroid_d
        
        # 3. 构建协方差矩阵并进行 SVD 分解
        H = np.dot(m_centered.T, d_centered)
        U, S, Vt = np.linalg.svd(H)
        
        # 4. 计算旋转矩阵 R
        R = np.dot(Vt.T, U.T)
        
        # 5. 计算平移向量 T
        T = centroid_d - np.dot(R, centroid_m)
        
        return R, T

    def apply_transform(self, measured, R, T):
        return np.dot(measured, R.T) + T