from math import sqrt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_excel('TP-Notes.xlsx')

class Tp:
    def __init__(self):
        self.note_exams = data['Examen/5']
        self.note_intero1 = data['Inter1/5']
        self.note_intero2 = data['Inter2/5']
        self.joint_i1_i2 = np.zeros((5,5))
        self.joint_i1_ex = np.zeros((5,5))
        self.joint_ex_i2 = np.zeros((5,5))
        self.joint_3d = np.zeros((5,5,5))
        self.note_intero1_interv = {'[0,1]':[],'[1,2]':[],'[2,3]':[],'[3,4]':[],'[4,5]':[]}
        self.note_intero2_interv = {'[0,1]':[],'[1,2]':[],'[2,3]':[],'[3,4]':[],'[4,5]':[]}
        self.note_exams_interv = {'[0,1]':[],'[1,2]':[],'[2,3]':[],'[3,4]':[],'[4,5]':[]}
        self.marginal_i1_i2 = np.zeros((6,6))
        self.marginal_i1_ex = np.zeros((6,6))
        self.marginal_ex_i2 = np.zeros((6,6))
        self.marginal_arr_i1 = []
        self.marginal_arr_i2 = []
        self.marginal_arr_ex = []
        self.cov_i1_i2 = np.zeros((2,2))
        self.cov_i1_ex = np.zeros((2,2))
        self.cov_ex_i2 = np.zeros((2,2))
        self.cov_3d = np.zeros((3,3))
        self.corr_i1_i2 = np.zeros((2,2))
        self.corr_i1_ex = np.zeros((2,2))
        self.corr_ex_i2 = np.zeros((2,2))
        self.corr_3d = np.zeros((3,3))

    def sep_inter(self): #1
        for i,j,m in zip(self.note_exams,self.note_intero1,self.note_intero2):
            if 0< i and i <= 1:
                self.note_exams_interv['[0,1]'].append(i)
            elif 1< i and i <= 2:
                self.note_exams_interv['[1,2]'].append(i)
            elif 2< i and i <= 3:
                self.note_exams_interv['[2,3]'].append(i)
            elif 3< i and i <= 4:
                self.note_exams_interv['[3,4]'].append(i)
            elif 4< i and i <= 5:
                self.note_exams_interv['[4,5]'].append(i)
            if 0< j and j <= 1:
                self.note_intero1_interv['[0,1]'].append(j)
            elif 1< j and j <= 2:
                self.note_intero1_interv['[1,2]'].append(j)
            elif 2< j and j <= 3:
                self.note_intero1_interv['[2,3]'].append(j)
            elif 3< j and j <= 4:
                self.note_intero1_interv['[3,4]'].append(j)
            elif 4< j and j <= 5:
                self.note_intero1_interv['[4,5]'].append(j)
            if 0< m and m <= 1:
                self.note_intero2_interv['[0,1]'].append(m)
            elif 1< m and m <= 2:
                self.note_intero2_interv['[1,2]'].append(m)
            elif 2< m and m <= 3:
                self.note_intero2_interv['[2,3]'].append(m)
            elif 3< m and m <= 4:
                self.note_intero2_interv['[3,4]'].append(m)
            elif 4< m and m <= 5:
                self.note_intero2_interv['[4,5]'].append(m)

    def joint_dist(self): #2
        for l in range(5):
            for i,j in zip(self.note_intero1,self.note_intero2):
                if l < i and i <= l+1:
                    if 0 < j and j <= 1:
                        self.joint_i1_i2[l, 0] += 1
                    elif 1 < j and j <= 2:
                        self.joint_i1_i2[l, 1] += 1
                    elif 2 < j and j <= 3:
                        self.joint_i1_i2[l, 2] += 1
                    elif 3 < j and j <= 4:
                        self.joint_i1_i2[l, 3] += 1
                    elif 4 < j and j <= 5:
                        self.joint_i1_i2[l, 4] += 1
            for i,j in zip(self.note_exams,self.note_intero2):
                if l < i and i <= l+1:
                    if 0 < j and j <= 1:
                        self.joint_ex_i2[l, 0] += 1
                    elif 1 < j and j <= 2:
                        self.joint_ex_i2[l, 1] += 1
                    elif 2 < j and j <= 3:
                        self.joint_ex_i2[l, 2] += 1
                    elif 3 < j and j <= 4:
                        self.joint_ex_i2[l, 3] += 1
                    elif 4 < j and j <= 5:
                        self.joint_ex_i2[l, 4] += 1
            for i,j in zip(self.note_intero1,self.note_exams):
                if l < i and i <= l+1:
                    if 0 < j and j <= 1:
                        self.joint_i1_ex[l, 0] += 1
                    elif 1 < j and j <= 2:
                        self.joint_i1_ex[l, 1] += 1
                    elif 2 < j and j <= 3:
                        self.joint_i1_ex[l, 2] += 1
                    elif 3 < j and j <= 4:
                        self.joint_i1_ex[l, 3] += 1
                    elif 4 < j and j <= 5:
                        self.joint_i1_ex[l, 4] += 1
        
        self.joint_i1_i2 /= 45
        self.joint_i1_ex /= 45
        self.joint_ex_i2 /= 45

    def joint_dist_3d(self): #3
        for l in range(5):
            for k in range(5):
                for i, m, j in zip(self.note_intero1,self.note_exams,self.note_intero2):
                    if l < i and i <= l+1 and k < m and m <= k+1:
                        if 0 < j and j <= 1:
                            self.joint_3d[0, k, l] += 1
                        elif 1 < j and j <= 2:
                            self.joint_3d[1, k, l] += 1
                        elif 2 < j and j <= 3:
                            self.joint_3d[2, k, l] += 1
                        elif 3 < j and j <= 4:
                            self.joint_3d[3, k, l] += 1
                        elif 4 < j and j <= 5:
                            self.joint_3d[4, k, l] += 1
        
        self.joint_3d /= 45

    def marginal_dist(self): #4
        sum_arr_row_i1i2 = []
        sum_arr_col_i1i2 = []
        sum_arr_row_exi2 = []
        sum_arr_col_exi2 = []
        sum_arr_row_i1ex = []
        sum_arr_col_i1ex = []

        for i in self.joint_i1_i2:
            sum_arr_row_i1i2.append(np.sum(i))
        for i in self.joint_i1_i2.T:
            sum_arr_col_i1i2.append(np.sum(i))
        for i in self.joint_ex_i2:
            sum_arr_row_exi2.append(np.sum(i))
        for i in self.joint_ex_i2.T:
            sum_arr_col_exi2.append(np.sum(i))
        for i in self.joint_i1_ex:
            sum_arr_row_i1ex.append(np.sum(i))
        for i in self.joint_i1_ex.T:
            sum_arr_col_i1ex.append(np.sum(i))

        for i in range(5):
            self.marginal_ex_i2[5,i] = sum_arr_col_exi2[i]
            self.marginal_ex_i2[i,5] = sum_arr_row_exi2[i]
            self.marginal_i1_ex[i,5] = sum_arr_row_i1ex[i]
            self.marginal_i1_ex[5,i] = sum_arr_col_i1ex[i]
            self.marginal_i1_i2[5,i] = sum_arr_col_i1i2[i]
            self.marginal_i1_i2[i,5] = sum_arr_row_i1i2[i]
            for j in range(5):
                self.marginal_ex_i2[i,j] = self.joint_ex_i2[i,j]
                self.marginal_i1_ex[i,j] = self.joint_i1_ex[i,j]
                self.marginal_i1_i2[i,j] = self.joint_i1_i2[i,j]
        self.marginal_i1_i2[5,5] = np.sum(sum_arr_col_i1i2)
        self.marginal_i1_ex[5,5] = np.sum(sum_arr_col_i1ex)
        self.marginal_ex_i2[5,5] = np.sum(sum_arr_col_exi2)

    def marginal_dist_3d(self): #5
        self.marginal_arr_i1 = self.marginal_i1_i2.T[5].copy()
        self.marginal_arr_i2 = self.marginal_i1_i2[5].copy()
        self.marginal_arr_ex = self.marginal_i1_ex[5].copy()
        self.marginal_arr_i1 = self.marginal_arr_i1[:-1]
        self.marginal_arr_i2 = self.marginal_arr_i2[:-1]
        self.marginal_arr_ex = self.marginal_arr_ex[:-1]

    def means(self, x):# 6
        Ni = np.array([0.5,1.5,2.5,3.5,4.5])
        return np.dot(Ni,x)
    def cov_vars(self, x,u, e1, e2):#7
        Ni = np.array([0.5,1.5,2.5,3.5,4.5])
        x1 = x-e1
        u1 = u-e2
        return (np.dot(Ni,(x1*u1))/ 5)
    def cov_var_mat(self):#8
        Exi1 = self.means(self.marginal_arr_i1)
        Exi2 = self.means(self.marginal_arr_i2)
        Exexam = self.means(self.marginal_arr_ex)
        Vari1 = self.cov_vars(self.marginal_arr_i1,self.marginal_arr_i1, Exi1,Exi1)
        Vari2 = self.cov_vars(self.marginal_arr_i2,self.marginal_arr_i2, Exi2,Exi2)
        Varexam = self.cov_vars(self.marginal_arr_ex,self.marginal_arr_ex, Exexam,Exexam)
        Covi1i2 = self.cov_vars(self.marginal_arr_i1,self.marginal_arr_i2, Exi1, Exi2)
        Covi1ex = self.cov_vars(self.marginal_arr_i1,self.marginal_arr_ex, Exi1, Exexam)
        Covexi2 = self.cov_vars(self.marginal_arr_ex,self.marginal_arr_i2, Exexam, Exi2)
        self.cov_i1_ex[0,0] = Vari1
        self.cov_i1_ex[1,1] = Varexam
        self.cov_i1_ex[1,0] = Covi1ex
        self.cov_i1_ex[0,1] = Covi1ex
        self.cov_ex_i2[0,0] = Vari2
        self.cov_ex_i2[1,1] = Varexam
        self.cov_ex_i2[1,0] = Covexi2
        self.cov_ex_i2[0,1] = Covexi2
        self.cov_i1_i2[0,0] = Vari1
        self.cov_i1_i2[1,1] = Vari2
        self.cov_i1_i2[1,0] = Covi1i2
        self.cov_i1_i2[0,1] = Covi1i2

    def cov_var_mat_3d(self):#9
        Exi1 = self.means(self.marginal_arr_i1)
        Exi2 = self.means(self.marginal_arr_i2)
        Exexam = self.means(self.marginal_arr_ex)
        Vari1 = self.cov_vars(self.marginal_arr_i1,self.marginal_arr_i1, Exi1,Exi1)
        Vari2 = self.cov_vars(self.marginal_arr_i2,self.marginal_arr_i2, Exi2,Exi2)
        Varexam = self.cov_vars(self.marginal_arr_ex,self.marginal_arr_ex, Exexam,Exexam)
        Covi1i2 = self.cov_vars(self.marginal_arr_i1,self.marginal_arr_i2, Exi1, Exi2)
        Covi1ex = self.cov_vars(self.marginal_arr_i1,self.marginal_arr_ex, Exi1, Exexam)
        Covexi2 = self.cov_vars(self.marginal_arr_ex,self.marginal_arr_i2, Exexam, Exi2)
        self.cov_3d[0,0] = Vari1
        self.cov_3d[0,1] = Covi1i2
        self.cov_3d[0,2] = Covi1ex
        self.cov_3d[1,0] = Covi1i2
        self.cov_3d[1,1] = Vari2
        self.cov_3d[1,2] = Covexi2
        self.cov_3d[2,0] = Covi1ex
        self.cov_3d[2,1] = Covexi2
        self.cov_3d[2,2] = Varexam
        
        
    def corr_mat(self):#10
        Exi1 = self.means(self.marginal_arr_i1)
        Exi2 = self.means(self.marginal_arr_i2)
        Exexam = self.means(self.marginal_arr_ex)
        Vari1 = self.cov_vars(self.marginal_arr_i1,self.marginal_arr_i1, Exi1,Exi1)
        Vari2 = self.cov_vars(self.marginal_arr_i2,self.marginal_arr_i2, Exi2,Exi2)
        Varexam = self.cov_vars(self.marginal_arr_ex,self.marginal_arr_ex, Exexam,Exexam)
        Covi1i2 = self.cov_vars(self.marginal_arr_i1,self.marginal_arr_i2, Exi1, Exi2)
        Covi1ex = self.cov_vars(self.marginal_arr_i1,self.marginal_arr_ex, Exi1, Exexam)
        Covexi2 = self.cov_vars(self.marginal_arr_ex,self.marginal_arr_i2, Exexam, Exi2)
        Stdi1 = sqrt(Vari1)
        Stdi2 = sqrt(Vari2)
        Stdexam = sqrt(Varexam)
        Corri1 = Vari1 / (Stdi1 * Stdi1)
        Corri2 = Vari2 / (Stdi2 * Stdi2)
        Correx = Varexam / (Stdexam * Stdexam)
        Corri1i2 = Covi1i2 / (Stdi2 * Stdi1)
        Corri1ex = Covi1ex / (Stdi1 * Stdexam)
        Corri2ex = Covexi2 / (Stdi2 * Stdexam)

        self.corr_ex_i2[0,0] = Correx
        self.corr_ex_i2[1,1] = Corri2
        self.corr_ex_i2[1,0] = Corri2ex
        self.corr_ex_i2[0,1] = Corri2ex

        self.corr_i1_ex[0,0] = Corri1
        self.corr_i1_ex[1,1] = Correx
        self.corr_i1_ex[1,0] = Corri1ex
        self.corr_i1_ex[0,1] = Corri1ex

        self.corr_i1_i2[0,0] = Corri1
        self.corr_i1_i2[1,1] = Corri2
        self.corr_i1_i2[1,0] = Corri1i2
        self.corr_i1_i2[0,1] = Corri1i2

    def corr_mat_3d(self): #11
        Exi1 = self.means(self.marginal_arr_i1)
        Exi2 = self.means(self.marginal_arr_i2)
        Exexam = self.means(self.marginal_arr_ex)
        Vari1 = self.cov_vars(self.marginal_arr_i1,self.marginal_arr_i1, Exi1,Exi1)
        Vari2 = self.cov_vars(self.marginal_arr_i2,self.marginal_arr_i2, Exi2,Exi2)
        Varexam = self.cov_vars(self.marginal_arr_ex,self.marginal_arr_ex, Exexam,Exexam)
        Covi1i2 = self.cov_vars(self.marginal_arr_i1,self.marginal_arr_i2, Exi1, Exi2)
        Covi1ex = self.cov_vars(self.marginal_arr_i1,self.marginal_arr_ex, Exi1, Exexam)
        Covexi2 = self.cov_vars(self.marginal_arr_ex,self.marginal_arr_i2, Exexam, Exi2)
        Stdi1 = sqrt(Vari1)
        Stdi2 = sqrt(Vari2)
        Stdexam = sqrt(Varexam)
        Corri1 = Vari1 / (Stdi1 * Stdi1)
        Corri2 = Vari2 / (Stdi2 * Stdi2)
        Correx = Varexam / (Stdexam * Stdexam)
        Corri1i2 = Covi1i2 / (Stdi2 * Stdi1)
        Corri1ex = Covi1ex / (Stdi1 * Stdexam)
        Corri2ex = Covexi2 / (Stdi2 * Stdexam)
        self.corr_3d[0,0] = Corri1
        self.corr_3d[0,1] = Corri1i2
        self.corr_3d[0,2] = Corri1ex
        self.corr_3d[1,0] = Corri1i2
        self.corr_3d[1,1] = Corri2
        self.corr_3d[1,2] = Corri2ex
        self.corr_3d[2,0] = Corri1ex
        self.corr_3d[2,1] = Corri2ex
        self.corr_3d[2,2] = Correx

    def independent(self): #12
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    if self.joint_3d[i][j][k] == (self.marginal_arr_i2[j] *self.marginal_arr_ex[k] *self.marginal_arr_i1[i]):
                        return 'independent'
    
        return 'dependent'

    def plot(self): #13
        fig = plt.figure(0,figsize=(8,8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.note_intero1,self.note_intero2, self.note_exams)
        ax.set_xlabel('Intero 1 Notes')
        ax.set_ylabel('Intero 2 Notes')
        ax.set_zlabel('Exams Notes')
        ax.set_title('Notes plot')
        plt.figure(1,figsize=(8,8))
        plt.xlabel('Intero1 Notes')
        plt.ylabel('Exams Notes')
        plt.title('Notes plot')
        plt.scatter(self.note_intero1, self.note_exams)
        plt.figure(2,figsize=(8,8))
        plt.xlabel('Intero2 Notes')
        plt.ylabel('Exams Notes')
        plt.title('Notes plot')
        plt.scatter(self.note_intero2, self.note_exams)
        plt.figure(2,figsize=(8,8))
        plt.xlabel('Intero2 Notes')
        plt.ylabel('Intero1 Notes')
        plt.title('Notes plot')
        plt.scatter(self.note_intero2, self.note_intero1)
        sns.jointplot(self.marginal_arr_i1, self.marginal_arr_i2)
        sns.jointplot(self.marginal_arr_i2, self.marginal_arr_ex)
        sns.jointplot(self.marginal_arr_i1, self.marginal_arr_ex)
        plt.show()
    

        

tp_object = Tp()
tp_object.sep_inter()
tp_object.joint_dist()
tp_object.joint_dist_3d()
tp_object.marginal_dist()
tp_object.marginal_dist_3d()
tp_object.cov_var_mat_3d()
tp_object.cov_var_mat()
tp_object.corr_mat()
tp_object.corr_mat_3d()
print(tp_object.independent())
tp_object.plot()
