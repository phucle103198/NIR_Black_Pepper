import numpy as np
from numpy import diag, cumsum, where, dot, outer, zeros, sqrt, mean, sum, min, square, inner
from numpy.linalg import inv, norm
import scipy.stats as sps
from scipy.spatial.distance import pdist, squareform


def polynomial_fit(x, y, order=1):  # 默认1次
    '''
    Least squares polynomial fit.
    :param x:
    :param y:
    :param order: 1,2,3...
    :return:
    regression_coefficient -
    fit_value -    fit_transform result (m X 1 column vector)
    residual -    residual   (m X 1 column vector)
    '''
    if x.ndim > 1:
        x = x.ravel()
    if y.ndim > 1:
        y = y.ravel()
    if len(x) != len(y):
        raise ValueError('The number of samples is not equal!')
    z = np.polyfit(x, y, deg=order)
    p = np.poly1d(z)
    fit_value = p(x)
    residual = fit_value - y
    regression_coefficient = z

    return {'regression_coefficient':regression_coefficient,
            'fit_value':fit_value,
            'residual':residual}



class MC(object):
    '''
    Mean Centering 均值中心化
    '''
    def __init__(self, avg_ab=None):
        self.avg_ab = avg_ab
        return

    def mc(self, spec, avg_ab=None):
        ab = spec[:, :]
        if avg_ab is None:
            avg_ab = np.mean(ab, axis=0)  # Get the mean of each column
        else:
            avg_ab = avg_ab
        ab_mc = ab - avg_ab  # 利用numpy数组的广播法则
        spec_mc = np.vstack((ab_mc))

        return spec_mc

    def fit(self, spec):
        self.wavelength = spec[0, :]
        ab = spec[1:, :]
        self.avg_ab = np.mean(ab, axis=0)
        return self

    def fit_transform(self, spec):
        self.wavelength = spec[0, :]
        ab = spec[1:, :]
        self.avg_ab = np.mean(ab, axis=0)
        spec_mc = self.mc(spec, avg_ab=self.avg_ab)
        return spec_mc

    def transform(self, input_data):
        '''
        用于当前实例
        :param input_data:
        :param avg_ab:
        :return:
        '''
        input_wavelength = input_data[0, :]
        spec_mc = self.mc(input_data, avg_ab=self.avg_ab)

        return spec_mc

    def inverse_transform(self, spec_mc, avg_ab=None):
        wavelength = spec_mc[0, :]
        ab_mc = spec_mc[1:, :]
        if avg_ab is None:
            ab_ori = ab_mc + self.avg_ab
        else:
            ab_ori = ab_mc + avg_ab
        spec_ori = np.vstack((wavelength, ab_ori))

        return spec_ori

class ZS(object):
    '''
    Zscore Standardization 中心标准化
    '''
    def __init__(self, avg_ab=None, std_ab=None):
        '''
        将原数据集各元素减去元素所在列的均值,再除以该列元素的标准差
        Centering using the average value, also called mean centering
        Scaling involves dividing the (centered) variables by individual measures of dispersion.
        Using the Standard Deviation as the scaling factor sets the variance for each variable to one,
        and is usually applied after mean centering.
        :param avg_ab:
        :param std_ab:
        '''
        self.avg_ab = avg_ab
        self.std_ab = std_ab
        return

    def zs(self, spec, avg_ab=None, std_ab=None):
        ab = spec[:, :]
        if avg_ab is None and std_ab is None:
            ab_mean = np.mean(ab, axis=0)  # Get the mean of each column
            ab_mc = ab - ab_mean
            stdev = np.std(ab_mc, axis=0, ddof=1)
            ab_zs = ab_mc / stdev
        elif avg_ab is not None and std_ab is not None:
            ab_mean = avg_ab
            ab_mc = ab - ab_mean
            stdev = std_ab
            ab_zs = ab_mc / stdev
        spec_zs = np.vstack(ab_zs)

        return spec_zs

    def fit(self, spec):
        ab = spec[:, :]
        self.ab_mean = np.mean(ab, axis=0)
        self.ab_std = np.std(ab, axis=0, ddof=1)
        return self

    def fit_transform(self, spec):
        ab = spec[:, :]
        self.ab_mean = np.mean(ab, axis=0)
        self.ab_std = np.std(ab, axis=0, ddof=1)
        spec_zs = self.zs(spec, avg_ab=self.ab_mean, std_ab=self.ab_std)
        return spec_zs

    def transform(self, input_data):
        spec_zs = self.zs(input_data, avg_ab=self.ab_mean, std_ab=self.ab_std)

        return spec_zs

    def inverse_transform(self, spec_as, avg_ab=None, std_ab=None):
        ab_zs = spec_as[:, :]
        if avg_ab is None and std_ab is None:
            ab_ori = ab_zs * self.ab_std + self.ab_mean
        else:
            ab_ori = ab_zs * std_ab + avg_ab
        spec_ori = np.vstack(( ab_ori))

        return spec_ori

class MSC(object):
    '''
    Multiplicative Scatter Correction 
    '''
    def __init__(self, ideal_ab=None):
        self.ideal_ab = ideal_ab
        return

    def msc(self, spec, ideal_ab=None):
        ab = spec[:, :]
        size_of_ab = ab.shape  # 10,700
        ab_msc = np.zeros(size_of_ab)  # 10,700

        if ideal_ab is None:
            ab_mean = np.mean(ab, axis=0)  # 700,
        elif len(ideal_ab) != len(np.mean(ab, axis=0)):
            raise ValueError('Error！')
        else:
            ab_mean = ideal_ab
        for i in range(size_of_ab[0]):  #  = b[0]   d = b[1]
            regression_coefficient = polynomial_fit(ab_mean, ab[i, :], order=1)['regression_coefficient']
            ab_msc[i, :] = (ab[i, :] - regression_coefficient[1]) / regression_coefficient[0]  # 
        spec_msc = np.vstack((ab_msc))

        return spec_msc

    def fit(self, spec):
        ab = spec[:, :]
        self.ideal_ab = np.mean(ab, axis=0)

        return self

    def fit_transform(self, spec):
        ab = spec[:, :]
        self.ideal_ab = np.mean(ab, axis=0)
        spec_msc = self.msc(spec, ideal_ab=self.ideal_ab)

        return spec_msc

    def transform(self, input_data):
        spec_msc = self.msc(input_data, ideal_ab=self.ideal_ab)
        return spec_msc

class SGMSC(object):
    '''
     Savitzky-Golay + Multiplicative Scatter Correction 一阶导 + 多元散射校正
    '''
    def __init__(self, window_size=11, polyorder=2, deriv=1, ideal_ab=None):
        self.window_size = window_size
        self.polyorder = polyorder
        self.deriv = deriv
        self.ideal_ab = ideal_ab

        return

    def _msc(self, spec, ideal_ab=None):
        ab = spec[:, :]
        size_of_ab = ab.shape  # 10,700
        ab_msc = np.zeros(size_of_ab)  # 10,700
        if ideal_ab is None:
            ab_mean = np.mean(ab, axis=0)  # 700,
        elif len(ideal_ab) != len(np.mean(ab, axis=0)):
            raise ValueError('Error！')
        else:
            ab_mean = ideal_ab
        d_add = np.ones(size_of_ab[1])  # 700,   线性偏移量offset
        matrix_A = (np.vstack((ab_mean, d_add))).T  # (700,2)
        for i in range(size_of_ab[0]):  # 求出每条光谱的c和d，c = b[0]   d = b[1]
            b = dot(dot(np.linalg.inv(dot(matrix_A.T, matrix_A)), matrix_A.T), ab[i, :])
            ab_msc[i, :] = (ab[i, :] - b[1]) / b[0]  # 利用广播法则
        spec_msc = np.vstack((ab_msc))

        return spec_msc

    def _sg(self, spec, window_size=11, polyorder=2, deriv=1):
        '''
        :param spec:
        :param window_size: must be odd and bigger than 2
        :param polyorder: must be bigger than deriv
        :param deriv:
        :return:
        '''
        try:
            window_size = np.abs(int(window_size))
            polyorder = np.abs(int(polyorder))
        except ValueError as msg:
            raise ValueError("window_size and polyorder have to be of type int")
        if window_size % 2 != 1 or window_size < 2:
            raise ValueError("window_size size must be a positive odd number")
        if window_size < polyorder:  # polyorder must be less than window_size
            raise ValueError("window_size is too small for the polynomials polyorder")
        if deriv > polyorder:  # 'deriv' must be less than or equal to 'polyorder'
            raise ValueError("Error!")

        n = spec.shape[0] - 1
        p = spec.shape[1]
        wavelength = spec[0, :]
        half_size = window_size // 2

        # 计算SG系数
        coef = np.zeros((window_size, polyorder + 1))
        for i in range(coef.shape[0]):
            for j in range(coef.shape[1]):
                coef[i, j] = np.power(i - int(window_size / 2), j)
        c = dot(inv(dot(coef.T, coef)), coef.T)

        # 拷贝SG系数
        coefs = np.zeros(window_size)  #(11,)
        for k in range(window_size):
            if deriv == 0 or deriv == 1:
                coefs[k] = c[deriv, k]
            elif deriv == 2:  # 需要调整系数
                coefs[k] = c[deriv, k] * 2
            elif deriv == 3:  # 需要调整系数
                coefs[k] = c[deriv, k] * 6
            elif deriv == 4:  # 需要调整系数
                coefs[k] = c[deriv, k] * 24

        # 处理吸光度
        tempdata = np.zeros((n, p))
        ab = spec[:, :]
        for j in range(0, p - window_size + 1):
            data_window = ab[:, j:j + window_size]
            new_y = dot(data_window, coefs[:, np.newaxis])  # 将coefs增加到二维
            tempdata[:, j + half_size] = new_y.ravel()

        # 处理两端的数据
        for j in range(0, half_size):
            tempdata[:, j] = tempdata[:, half_size]
        for j in range(p - half_size, p):
            tempdata[:, j] = tempdata[:, p - half_size - 1]

        # 导数
        if deriv > 0:
            x_step = wavelength[1] - wavelength[0]
            x_step = np.power(x_step, deriv)
            ab_sg = tempdata / x_step
        else:
            ab_sg = tempdata

        spec_sg_matrix = np.vstack((ab_sg))
        return spec_sg_matrix

    def sgmsc(self, spec, window_size=11, polyorder=2, deriv=1, ideal_ab=None):
        spec_sg = self._sg(spec, window_size=window_size, polyorder=polyorder, deriv=deriv)
        spec_sg_msc = self._msc(spec_sg, ideal_ab=ideal_ab)
        return spec_sg_msc

    def fit(self, spec):
        self.wavelength = spec[0, :]
        self.ideal_ab = np.mean(spec[:, :], axis=0)
        return self

    def fit_transform(self, spec):
        spec_sg = self._sg(spec, window_size=self.window_size, polyorder=self.polyorder, deriv=self.deriv)
        self.fit(spec_sg)
        spec_sg_msc = self._msc(spec_sg, ideal_ab=self.ideal_ab)

        return spec_sg_msc

    def transform(self, input_data):
        spec_sg = self._sg(input_data, window_size=self.window_size, polyorder=self.polyorder,
                                   deriv=self.deriv)
        spec_sg_msc = self._msc(spec_sg, ideal_ab=self.ideal_ab)

        return spec_sg_msc

# -------- 单样本操作 --------
class VN(object):
    '''
    Vector Normalization矢量归一化
    '''
    def __init__(self):
        return

    def vn(self, spec):
        ab = spec[0:, :]
        ab_vn = ab / np.linalg.norm(ab, axis=1, keepdims=True)
        spec_vn = np.vstack((ab_vn))
        return spec_vn

    def fit_transform(self, spec):
        spec_vn = self.vn(spec)
        return spec_vn

    def transform(self, input_data):
        spec_vn = self.vn(input_data)
        return spec_vn

class SNV(object):
    '''
    Standard Normal Variate transformation 
    '''
    def __init__(self):
        return

    def snv(self, spec):
        ab = spec[:, :]
        ab_mc = ab - np.mean(ab, axis=1, keepdims=True)
        ab_snv = ab_mc / np.std(ab, axis=1, keepdims=True, ddof=1)
        spec_snv = np.vstack((ab_snv))
        return spec_snv

    def fit_transform(self, spec):
        spec_snv = self.snv(spec)
        return spec_snv

    def transform(self, input_data):
        spec_vn = self.snv(input_data)
        return spec_vn

class ECO(object):
    '''
    Eliminate Constant Offset 消除常数偏移量(减去各条光谱的最小值，使得最小值变成0)
    '''
    def __init__(self):
        return

    def eco(self, spec):
        wavelength = spec[0, :]
        ab = spec[1:, :]
        ab_sco = ab - np.min(ab, axis=1, keepdims=True)
        spec_sco = np.vstack((wavelength, ab_sco))
        return spec_sco

    def fit_transform(self, spec):
        self.wavelength = spec[0, :]
        spec_eco = self.eco(spec)
        return spec_eco

    def transform(self, input_data):
        input_wavelength = input_data[0, :]
        if not (input_wavelength == self.wavelength).all():
            raise ValueError('光谱数据不兼容！')
        spec_eco = self.eco(input_data)
        return spec_eco

class SSL(object):
    '''
    Subtract Straight Line 减去一条直线
    '''
    def __init__(self):
        return

    def ssl(self, spec):  # 必须含有波长
        wavelength = spec[0, :]
        ab = spec[1:, :]
        n_samples = ab.shape[0]
        ab_ssl = np.zeros(ab.shape)
        for i in range(n_samples):  # 求出趋势直线
            fit_value = polynomial_fit(wavelength, ab[i, :], order=1)['fit_value']
            ab_ssl[i, :] = ab[i, :] - fit_value.ravel()
        spec_ssl = np.vstack((wavelength, ab_ssl))

        return spec_ssl

    def fit_transform(self, spec):
        self.wavelength = spec[0, :]
        spec_ssl = self.ssl(spec)

        return spec_ssl

    def transform(self, input_data):
        input_wavelength = input_data[0, :]
        if not (input_wavelength == self.wavelength).all():
            raise ValueError('光谱数据不兼容！')
        spec_ssl = self.ssl(input_data)
        return spec_ssl

class DT(object):
    '''
    De-Trending 去趋势(2次多项式)
    '''
    def __init__(self):
        return

    def dt(self, spec):  # 必须含有波长
        wavelength = spec[0, :]
        ab = spec[1:, :]
        n_samples = ab.shape[0]
        ab_dt = np.zeros(ab.shape)
        for i in range(n_samples):  # 求出每条光谱的c和d，c = b[0]   d = b[1]
            fit_value = polynomial_fit(wavelength, ab[i, :], order=2)['fit_value']
            ab_dt[i, :] = ab[i, :] - fit_value.ravel()
        spec_dt = np.vstack((wavelength, ab_dt))

        return spec_dt

    def fit_transform(self, spec):
        self.wavelength = spec[0, :]
        spec_dt = self.dt(spec)

        return spec_dt

    def transform(self, input_data):
        input_wavelength = input_data[0, :]
        if not (input_wavelength == self.wavelength).all():
            raise ValueError('光谱数据不兼容！')
        spec_dt = self.dt(input_data)
        return spec_dt

class MMN(object):  # just used for spectra preprocessing
    '''
    Min-Max Normalization 最小最大归一化
    '''
    def __init__(self, norm_min=0, norm_max=1):
        self.norm_min = norm_min
        self.norm_max = norm_max
        return

    def mmn(self, spec, norm_min=0, norm_max=1):  # min max normalize
        wavelength = spec[0, :]
        ab = spec[1:, :]
        xmin = np.min(ab, axis=1, keepdims=True)
        xmax = np.max(ab, axis=1, keepdims=True)
        ab_mmn = norm_min + (ab - xmin) * (norm_max - norm_min) / (xmax - xmin)
        spec_mmn = np.vstack((wavelength, ab_mmn))
        return spec_mmn

    def fit_transform(self, spec):
        self.wavelength = spec[0, :]
        spec_mmn = self.mmn(spec, norm_min=self.norm_min, norm_max=self.norm_max)
        return spec_mmn

    def transform(self, input_data):
        input_wavelength = input_data[0, :]
        if not (input_wavelength == self.wavelength).all():
            raise ValueError('光谱数据不兼容！')
        spec_mmn = self.mmn(input_data, norm_min=self.norm_min, norm_max=self.norm_max)
        return spec_mmn

class SG(object):
    '''
    Savitzky-Golay 平滑与求导
    '''
    def __init__(self, window_size=11, polyorder=2, deriv=1):
        '''
        OPUS中polyorder默认为2
        :param window_size:
        :param polyorder:
        :param deriv:
        '''
        self.window_size = window_size
        self.polyorder = polyorder
        self.deriv = deriv
        return

    def sg(self, spec, window_size=11, polyorder=2, deriv=1):
        '''

        :param spec:
        :param window_size: must be odd and bigger than 2
        :param polyorder: must be bigger than deriv
        :param deriv:
        :return:
        '''
        # ============ check parameters ============
        if not isinstance(window_size, int) or not isinstance(polyorder, int) or not isinstance(deriv, int):
            raise ValueError("'window_size', 'polyorder' and 'deriv' must be of type int")
        if window_size < 0 or polyorder < 0 or deriv < 0:
            raise ValueError("'window_size', 'polyorder' and 'deriv' cannot be less than 0")
        if window_size % 2 != 1 or window_size < 3:
            raise ValueError("'window_size' must be a positive odd number and greater than or equal to 3")
        if polyorder > window_size - 1:  # polyorder must be less than window_size
            raise ValueError("'polyorder' must be less than 'window_size'")
        if deriv > polyorder:  # deriv must be less than polyorder
            raise ValueError("'deriv' must be less than or equal to 'polyorder'")

        n = spec.shape[0] - 1
        p = spec.shape[1]
        half_size = window_size // 2

        # 计算SG系数
        coef = np.zeros((window_size, polyorder + 1))
        for i in range(coef.shape[0]):
            for j in range(coef.shape[1]):
                coef[i, j] = np.power(i - int(window_size / 2), j)
        c = dot(inv(dot(coef.T, coef)), coef.T)

        # 拷贝SG系数
        coefs = np.zeros(window_size)  #(11,)
        for k in range(window_size):
            if deriv == 0 or deriv == 1:
                coefs[k] = c[deriv, k]
            elif deriv == 2:  # 需要调整系数
                coefs[k] = c[deriv, k] * 2
            elif deriv == 3:  # 需要调整系数
                coefs[k] = c[deriv, k] * 6
            elif deriv == 4:  # 需要调整系数
                coefs[k] = c[deriv, k] * 24

        # 处理吸光度
        tempdata = np.zeros((n, p))
        ab = spec[1:, :]
        for j in range(0, p - window_size + 1):
            data_window = ab[:, j:j + window_size]
            new_y = dot(data_window, coefs[:, np.newaxis])  # 将coefs增加到二维
            tempdata[:, j + half_size] = new_y.ravel()

        # 处理两端的数据
        for j in range(0, half_size):
            tempdata[:, j] = tempdata[:, half_size]
        for j in range(p - half_size, p):
            tempdata[:, j] = tempdata[:, p - half_size - 1]

        # 导数
        if deriv > 0:
            x_step = wavelength[1] - wavelength[0]
            x_step = np.power(x_step, deriv)
            ab_sg = tempdata / x_step
        else:
            ab_sg = tempdata

        spec_sg_matrix = np.vstack((ab_sg))

        return spec_sg_matrix

    def fit_transform(self, spec, wavelength):
        self.wavelength = wavelength
        spec_sg = self.sg(spec, self.window_size, self.polyorder, self.deriv)
        return spec_sg

    def transform(self, input_data, wavelength):
        input_wavelength = wavelength
        if not (input_wavelength == self.wavelength).all():
            raise ValueError('光谱数据不兼容！')
        spec_sg = self.sg(input_data, self.window_size, self.polyorder, self.deriv, waveength)
        return spec_sg

class SGSNV(object):
    '''
    Savitzky-Golay + SNV
    '''
    def __init__(self, window_size=11, polyorder=2, deriv=1):
        self.window_size = window_size
        self.polyorder = polyorder
        self.deriv = deriv
        return

    def _snv(self, spec):
        wavelength = spec[0, :]
        ab = spec[1:, :]
        ab_snv = (ab - np.mean(ab, axis=1, keepdims=True)) / np.std(ab, axis=1, keepdims=True, ddof=1)
        spec_snv = np.vstack((wavelength, ab_snv))
        return spec_snv

    def _sg(self, spec, window_size=11, polyorder=2, deriv=1):
        '''

        :param spec:
        :param window_size: must be odd and bigger than 2
        :param polyorder: must be bigger than deriv
        :param deriv:
        :return:
        '''
        # ============ check parameters ============
        if not isinstance(window_size, int) or not isinstance(polyorder, int) or not isinstance(deriv, int):
            raise ValueError("'window_size', 'polyorder' and 'deriv' must be of type int")
        if window_size < 0 or polyorder < 0 or deriv < 0:
            raise ValueError("'window_size', 'polyorder' and 'deriv' cannot be less than 0")
        if window_size % 2 != 1 or window_size < 3:
            raise ValueError("'window_size' must be a positive odd number and greater than or equal to 3")
        if polyorder > window_size - 1:  # polyorder must be less than window_size
            raise ValueError("'polyorder' must be less than 'window_size'")
        if deriv > polyorder:  # deriv must be less than polyorder
            raise ValueError("'deriv' must be less than or equal to 'polyorder'")

        n = spec.shape[0] - 1
        p = spec.shape[1]
        wavelength = spec[0, :]
        half_size = window_size // 2

        # 计算SG系数
        coef = np.zeros((window_size, polyorder + 1))
        for i in range(coef.shape[0]):
            for j in range(coef.shape[1]):
                coef[i, j] = np.power(i - int(window_size / 2), j)
        c = dot(inv(dot(coef.T, coef)), coef.T)

        # 拷贝SG系数
        coefs = np.zeros(window_size)  #(11,)
        for k in range(window_size):
            if deriv == 0 or deriv == 1:
                coefs[k] = c[deriv, k]
            elif deriv == 2:  # 需要调整系数
                coefs[k] = c[deriv, k] * 2
            elif deriv == 3:  # 需要调整系数
                coefs[k] = c[deriv, k] * 6
            elif deriv == 4:  # 需要调整系数
                coefs[k] = c[deriv, k] * 24

        # 处理吸光度
        tempdata = np.zeros((n, p))
        ab = spec[1:, :]
        for j in range(0, p - window_size + 1):
            data_window = ab[:, j:j + window_size]
            new_y = dot(data_window, coefs[:, np.newaxis])  # 将coefs增加到二维
            tempdata[:, j + half_size] = new_y.ravel()

        # 处理两端的数据
        for j in range(0, half_size):
            tempdata[:, j] = tempdata[:, half_size]
        for j in range(p - half_size, p):
            tempdata[:, j] = tempdata[:, p - half_size - 1]

        # 导数
        if deriv > 0:
            x_step = wavelength[1] - wavelength[0]
            x_step = np.power(x_step, deriv)
            ab_sg = tempdata / x_step
        else:
            ab_sg = tempdata

        spec_sg_matrix = np.vstack((wavelength, ab_sg))

        return spec_sg_matrix

    def sgsnv(self, spec, window_size=11, polyorder=2, deriv=1):
        spec_sg = self._sg(spec, window_size=window_size, polyorder=polyorder, deriv=deriv)
        spec_sg_snv = self._snv(spec_sg)

        return spec_sg_snv

    def fit_transform(self, spec):
        self.wavelength = spec[0, :]
        spec_sg = self._sg(spec, window_size=self.window_size, polyorder=self.polyorder, deriv=self.deriv)
        spec_sg_snv = self._snv(spec_sg)

        return spec_sg_snv

    def transform(self, input_data):
        input_wavelength = input_data[0, :]
        if not (input_wavelength == self.wavelength).all():
            raise ValueError('光谱数据不兼容！')
        spec_sg = self._sg(input_data, window_size=self.window_size, polyorder=self.polyorder, deriv=self.deriv)
        spec_sg_snv = self._snv(spec_sg)

        return spec_sg_snv

class SNVDT(object):
    '''
    SNV + DT
    '''
    def __init__(self):

        return

    def _snv(self, spec):
        wavelength = spec[0, :]
        ab = spec[1:, :]
        ab_snv = (ab - np.mean(ab, axis=1, keepdims=True)) / np.std(ab, axis=1, keepdims=True, ddof=1)
        spec_snv = np.vstack((wavelength, ab_snv))

        return spec_snv

    def _dt(self, spec):  # 必须含有波长
        wavelength = spec[0, :]
        ab = spec[1:, :]
        n_samples = ab.shape[0]
        ab_dt = np.zeros(ab.shape)
        for i in range(n_samples):  # 求出每条光谱的c和d，c = b[0]   d = b[1]
            fit_value = polynomial_fit(wavelength, ab[i, :], order=2)['fit_value']
            ab_dt[i, :] = ab[i, :] - fit_value.ravel()
        spec_dt = np.vstack((wavelength, ab_dt))

        return spec_dt

    def snvdt(self, spec):
        spec_snv = self._snv(spec)
        spec_snv_dt = self._dt(spec_snv)

        return spec_snv_dt

    def fit_transform(self, spec):
        self.wavelength = spec[0, :]
        spec_snv = self._snv(spec)
        spec_snv_dt = self._dt(spec_snv)

        return spec_snv_dt

    def transform(self, input_data):
        input_wavelength = input_data[0, :]
        if not (input_wavelength == self.wavelength).all():
            raise ValueError('光谱数据不兼容！')
        spec_snv = self._snv(input_data)
        spec_snv_dt = self._dt(spec_snv)

        return spec_snv_dt

class SGSSL(object):
    '''
    SG + SSL  求导 + 减去一条直线
    '''
    def __init__(self, window_size=11, polyorder=2, deriv=1):
        self.window_size = window_size
        self.polyorder = polyorder
        self.deriv = deriv
        return

    def _ssl(self, spec):  # 必须含有波长
        size_of_spec = spec.shape  # 第一行是x轴
        wavelength = spec[0, :]
        spec_ssl = np.zeros(size_of_spec)
        spec_ssl[0, :] = wavelength
        f_add = np.ones(size_of_spec[1])  # 用于构造A
        matrix_A = (np.vstack((wavelength, f_add))).T  # 2126 * 2
        for i in range(1, size_of_spec[0]):  # 从1开始，不算wavelength
            r = dot(dot(np.linalg.inv(dot(matrix_A.T, matrix_A)), matrix_A.T), spec[i, :])
            spec_ssl[i, :] = spec[i, :] - dot(matrix_A, r)
        return spec_ssl

    def _sg(self, spec, window_size=11, polyorder=2, deriv=1):
        '''

        :param spec:
        :param window_size: must be odd and bigger than 2
        :param polyorder: must be bigger than deriv
        :param deriv:
        :return:
        '''
        # ============ check parameters ============
        if not isinstance(window_size, int) or not isinstance(polyorder, int) or not isinstance(deriv, int):
            raise ValueError("'window_size', 'polyorder' and 'deriv' must be of type int")
        if window_size < 0 or polyorder < 0 or deriv < 0:
            raise ValueError("'window_size', 'polyorder' and 'deriv' cannot be less than 0")
        if window_size % 2 != 1 or window_size < 3:
            raise ValueError("'window_size' must be a positive odd number and greater than or equal to 3")
        if polyorder > window_size - 1:  # polyorder must be less than window_size
            raise ValueError("'polyorder' must be less than 'window_size'")
        if deriv > polyorder:  # deriv must be less than polyorder
            raise ValueError("'deriv' must be less than or equal to 'polyorder'")

        n = spec.shape[0] - 1
        p = spec.shape[1]
        wavelength = spec[0, :]
        half_size = window_size // 2

        # 计算SG系数
        coef = np.zeros((window_size, polyorder + 1))
        for i in range(coef.shape[0]):
            for j in range(coef.shape[1]):
                coef[i, j] = np.power(i - int(window_size / 2), j)
        c = dot(inv(dot(coef.T, coef)), coef.T)

        # 拷贝SG系数
        coefs = np.zeros(window_size)  #(11,)
        for k in range(window_size):
            if deriv == 0 or deriv == 1:
                coefs[k] = c[deriv, k]
            elif deriv == 2:  # 需要调整系数
                coefs[k] = c[deriv, k] * 2
            elif deriv == 3:  # 需要调整系数
                coefs[k] = c[deriv, k] * 6
            elif deriv == 4:  # 需要调整系数
                coefs[k] = c[deriv, k] * 24

        # 处理吸光度
        tempdata = np.zeros((n, p))
        ab = spec[1:, :]
        for j in range(0, p - window_size + 1):
            data_window = ab[:, j:j + window_size]
            new_y = dot(data_window, coefs[:, np.newaxis])  # 将coefs增加到二维
            tempdata[:, j + half_size] = new_y.ravel()

        # 处理两端的数据
        for j in range(0, half_size):
            tempdata[:, j] = tempdata[:, half_size]
        for j in range(p - half_size, p):
            tempdata[:, j] = tempdata[:, p - half_size - 1]

        # 导数
        if deriv > 0:
            x_step = wavelength[1] - wavelength[0]
            x_step = np.power(x_step, deriv)
            ab_sg = tempdata / x_step
        else:
            ab_sg = tempdata

        spec_sg_matrix = np.vstack((wavelength, ab_sg))

        return spec_sg_matrix

    def sgssl(self, spec, window_size=11, polyorder=2, deriv=1):
        spec_sg = self._sg(spec, window_size=window_size, polyorder=polyorder, deriv=deriv)
        spec_sg_ssl = self._ssl(spec_sg)

        return spec_sg_ssl

    def fit_transform(self, spec):
        self.wavelength = spec[0, :]
        spec_sg = self._sg(spec, window_size=self.window_size, polyorder=self.polyorder, deriv=self.deriv)
        spec_sg_ssl = self._ssl(spec_sg)

        return spec_sg_ssl

    def transform(self, input_data):
        input_wavelength = input_data[0, :]
        if not (input_wavelength == self.wavelength).all():
            raise ValueError('光谱数据不兼容！')
        spec_sg = self._sg(input_data, window_size=self.window_size, polyorder=self.polyorder, deriv=self.deriv)
        spec_sg_ssl = self._ssl(spec_sg)

        return spec_sg_ssl

