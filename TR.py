import numpy as np
import soundfile as sf
from scipy.signal import hilbert
from statistics import mean
import matplotlib.pyplot as plt
from scipy.signal import butter
import scipy as sp
import pandas as pd


def mediamovil(signal, M):
    # Filtro de media movil de implementación directa. M es el número de puntos en el promedio
    M = M
    """
    xfd = np.zeros(len(signal)-M)
    for i in range(len(xfd)):
        for k in range(M):
            z = signal[i+k]
            xfd[i] = xfd[i]+z

    xfd = xfd*(1/(M+1))  # señal filtrada
    """
    xfr = np.zeros(len(signal)-M)
    for i in range(M+1):
        xfr[0] += signal[i]
    k = 0
    for i in range(len(signal)-M-1):
        xfr[i+1] = xfr[i] + signal[M+k+1]-signal[k]
        k += 1

    xfr = xfr*(1/(M+1))

    return xfr, M


def cuadradosminimos(x, y):
    # Obtener m y b de la recta de cuadrados mínimos y el valor r (relación lineal entre las variables aleatorias)
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    xy = np.multiply(x, y)
    x_sqr = x**2
    y_sqr = y**2
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum(xy)
    n = np.size(x)
    sum_xsqr = sum(x_sqr)
    sum_ysqr = sum(y_sqr)
    y_prom = mean(y)
    x_prom = mean(x)
    m = (n*sum_xy - sum_x*sum_y)/(n*sum_xsqr-sum_x**2)
    b = (sum_y - m*sum_x)/n
    r = (sum_xy - n*x_prom*y_prom)/(np.sqrt((sum_xsqr - n*(x_prom**2))*(sum_ysqr - n*(y_prom**2))))

    return b, m


def pasabanda_butter(lowcut, highcut, fs, order):
    """ Diseña un filtro pasabanda a partir de especificar las frecuencias
    de corte superior e inferior, la frecuencia de sampleo y el orden del filtro """
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    sos = butter(order, [low, high], btype='band', analog=False, output='sos')
    return sos


def filtrar(data, lowcut, highcut, fs, order):
    """ filtra una señal (data) a partir de diseñar un filtro pasabanda
    tipo butter a partir de las frecuencias de corte superior e inferior
    la frecuencia de sampleo y el orden del filtro. Devuelve la señal filtrada
    en tiempo """
    sos = pasabanda_butter(lowcut, highcut, fs, order)
    y = sp.signal.sosfilt(sos, data)
    return y


def TR(signal, fs, M):
    # Defino una matriz para guardar alli lo que obtenga de cada filtro
    # Msalida = np.zeros([10, len(signal)])
    Msalida = np.zeros([7, len(signal)])

    # A partir de los centros de bandas nominales (norma) defino las frecuencias de corte superiores e inferiores
    # y las almaceno en una lista para luego usarla cuando itere filtrando la señal
    # centros = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
    centros = [125, 250, 500, 1000, 2000, 4000, 8000]
    cortes = []
    n = 1  # bandas de octava
    order = 6  # Orden del filtro
    TR_SCH, TR_MMv = [], []

    for i in centros:
        fsup = i*np.power(2, 1/(2*n))
        finf = i*np.power(2, (-1)/(2*n))
        cortes.append([finf, fsup])

    # Para cada par de frecuencias de corte, diseño el filtro, filtro la señal, la convierto a dB
    # y calculo el nivel equivalente sin ponderar. Almaceno este valor final en una lista

    EDT_fin_MMv, EDT_fin_SCH = [], []
    T10_fin_MMv, T10_fin_SCH = [], []
    T20_fin_MMv, T20_fin_SCH = [], []
    T20_fin_MMv, T20_fin_SCH = [], []
    T30_fin_MMv, T30_fin_SCH = [], []
    T30_fin_MMv, T30_fin_SCH = [], []

    for i in cortes:
        lowcut, highcut = i[0], i[1]
        sig_f = filtrar(signal, lowcut, highcut, fs, order=order)

        signal_e = hilbert(sig_f)  # Lo suavizo mediante la transformada de Hilbert
        signal_e = np.abs(signal_e)

        for t in range(len(sig_f)):  # Me saco del medio el problemita de que el log de 0 da -infinito
            if signal_e[t] == 0:
                signal_e[t] = 0.0000001 #podría mejorarlo poniendo el número más bajo permitido

        signal_edB = 20*np.log10(signal_e/np.max(signal_e))  # Paso a dBFS

        # Cálculo del TR mediante el suavizado de un filtro de media movil
        TR_MMv, M = mediamovil(signal_edB, M)
        max = np.max(TR_MMv)
        for t in range(len(TR_MMv)):
            TR_MMv[t] -= max

        a = int(np.where(TR_MMv == 0)[0])  # Corto la señal en el máximo
        TR_MMv = TR_MMv[a:]

        # Determino un corte de la señal para poder hacer cuadrados mínimos con lo que me sirve
        b = np.where(TR_MMv < -45)[0][0]
        TR_MMv = TR_MMv[:b]

        # Cálculo del TR mediante suavizado por schroeder
        H_0 = signal_e[a:]
        H_0 = H_0[:b]
        H = np.flip(H_0)  # Hilbert recortado según los límites del de media movil
        SCH = np.cumsum(H**2)  # Transformada de Schroeder
        SCH = np.flip(SCH)

        for t in range(len(SCH)):  # Me saco del medio el problemita de que el log de 0 da -infinito
            if SCH[t] == 0:
                SCH[t] = 0.0000001

        SCHdB = 10*np.log10(SCH/np.max(SCH))  # Paso a dBFS

        # Corto la señal original para comparar:
        signal_edB = signal_edB[a:]
        signal_edB = signal_edB[:b]

        # Cálculo del tr:
        # EDT:
        x_last_MMv_EDT = np.where(TR_MMv < -10)[0][0]
        x_last_SCH_EDT = np.where(SCHdB < -10)[0][0]

        y_MMv_EDT = TR_MMv[0:x_last_MMv_EDT]
        x_MMv_EDT = np.arange(0, x_last_MMv_EDT, 1).tolist()

        b_MMv_EDT, m_MMv_EDT = cuadradosminimos(x_MMv_EDT, y_MMv_EDT)
        EDT_MMv = (-60-b_MMv_EDT)/(m_MMv_EDT*fs)

        y_SCH_EDT = SCHdB[0:x_last_SCH_EDT]
        x_SCH_EDT = np.arange(0, x_last_SCH_EDT, 1).tolist()

        b_SCH_EDT, m_SCH_EDT = cuadradosminimos(x_SCH_EDT, y_SCH_EDT)
        EDT_SCH = (-60-b_SCH_EDT)/(m_SCH_EDT*fs)

        # T10:
        x_first_MMv_T10 = np.where(TR_MMv < -5)[0][0]
        x_first_SCH_T10 = np.where(SCHdB < -5)[0][0]
        x_last_MMv_T10 = np.where(TR_MMv < -15)[0][0]
        x_last_SCH_T10 = np.where(SCHdB < -15)[0][0]

        y_MMv_T10 = TR_MMv[x_first_MMv_T10:x_last_MMv_T10]
        x_MMv_T10 = np.arange(x_first_MMv_T10, x_last_MMv_T10, 1).tolist()

        b_MMv_T10, m_MMv_T10 = cuadradosminimos(x_MMv_T10, y_MMv_T10)
        T10_MMv = (-60-b_MMv_T10)/(m_MMv_T10*fs)

        y_SCH_T10 = SCHdB[x_first_SCH_T10:x_last_SCH_T10]
        x_SCH_T10 = np.arange(x_first_SCH_T10, x_last_SCH_T10, 1).tolist()

        b_SCH_T10, m_SCH_T10 = cuadradosminimos(x_SCH_T10, y_SCH_T10)
        T10_SCH = (-60-b_SCH_T10)/(m_SCH_T10*fs)

        # T20:
        x_first_MMv_T20 = np.where(TR_MMv < -5)[0][0]
        x_first_SCH_T20 = np.where(SCHdB < -5)[0][0]
        x_last_MMv_T20 = np.where(TR_MMv < -25)[0][0]
        x_last_SCH_T20 = np.where(SCHdB < -25)[0][0]

        y_MMv_T20 = TR_MMv[x_first_MMv_T20:x_last_MMv_T20]
        x_MMv_T20 = np.arange(x_first_MMv_T20, x_last_MMv_T20, 1).tolist()

        b_MMv_T20, m_MMv_T20 = cuadradosminimos(x_MMv_T20, y_MMv_T20)
        T20_MMv = (-60-b_MMv_T20)/(m_MMv_T20*fs)

        y_SCH_T20 = SCHdB[x_first_SCH_T20:x_last_SCH_T20]
        x_SCH_T20 = np.arange(x_first_SCH_T20, x_last_SCH_T20, 1).tolist()

        b_SCH_T20, m_SCH_T20 = cuadradosminimos(x_SCH_T20, y_SCH_T20)
        T20_SCH = (-60-b_SCH_T20)/(m_SCH_T20*fs)

        # T30:
        x_first_MMv_T30 = np.where(TR_MMv < -5)[0][0]
        x_first_SCH_T30 = np.where(SCHdB < -5)[0][0]
        x_last_MMv_T30 = np.where(TR_MMv < -35)[0][0]
        x_last_SCH_T30 = np.where(SCHdB < -35)[0][0]

        y_MMv_T30 = TR_MMv[x_first_MMv_T30:x_last_MMv_T30]
        x_MMv_T30 = np.arange(x_first_MMv_T30, x_last_MMv_T30, 1).tolist()

        b_MMv_T30, m_MMv_T30 = cuadradosminimos(x_MMv_T30, y_MMv_T30)
        T30_MMv = (-60-b_MMv_T30)/(m_MMv_T30*fs)

        y_SCH_T30 = SCHdB[x_first_SCH_T30:x_last_SCH_T30]
        x_SCH_T30 = np.arange(x_first_SCH_T30, x_last_SCH_T30, 1).tolist()

        b_SCH_T30, m_SCH_T30 = cuadradosminimos(x_SCH_T30, y_SCH_T30)
        T30_SCH = (-60-b_SCH_T30)/(m_SCH_T30*fs)

        EDT_fin_MMv.append(EDT_MMv)
        EDT_fin_SCH.append(EDT_SCH)
        T10_fin_MMv.append(T10_MMv)
        T10_fin_SCH.append(T10_SCH)
        T20_fin_MMv.append(T20_MMv)
        T20_fin_SCH.append(T20_SCH)
        T30_fin_MMv.append(T30_MMv)
        T30_fin_SCH.append(T30_SCH)

    return EDT_fin_MMv, EDT_fin_SCH, T10_fin_MMv, T10_fin_SCH, T20_fin_MMv, T20_fin_SCH, T30_fin_MMv, T30_fin_SCH

"""

Cargo el audio y defino los parámetros!!!!

"""
signal, fs = sf.read('RESPUESTA A 2.wav')  # Cargo audio
M = 501  # Cantidad de promediaciones del filtro de media movil


# Cálculo global (sin filtrar)

signal_e = hilbert(signal)  # Lo suavizo mediante la transformada de Hilbert
signal_e = np.abs(signal_e)

for i in range(len(signal)):  # Me saco del medio el problemita de que el log de 0 da -infinito
    if signal_e[i] == 0:
        signal_e[i] = 0.0000001

signal_edB = 20*np.log10(signal_e/np.max(signal_e))  # Paso a dBFS


# Cálculo del TR mediante el suavizado de un filtro de media movil
TR_MMv, M = mediamovil(signal_edB, M)
max = np.max(TR_MMv)
for i in range(len(TR_MMv)):
    TR_MMv[i] -= max

print('Corte inferior:', np.where(TR_MMv == 0)[0][0])
a = int(np.where(TR_MMv == 0)[0])  # Corto la señal en el máximo
TR_MMv = TR_MMv[a:]
print('Corte superior:', np.where(TR_MMv < -45)[0][0])
# Determino un corte de la señal para poder hacer cuadrados mínimos con lo que me sirve
b = np.where(TR_MMv < -45)[0][0]
TR_MMv = TR_MMv[:b]

# Cálculo del TR mediante suavizado por schroeder
H_0 = signal_e[a:]
H_0 = H_0[:b]
H = np.flip(H_0)  # Hilbert recortado según los límites del de media movil
SCH = np.cumsum(H**2)  # Transformada de Schroeder
SCH = np.flip(SCH)


for i in range(len(SCH)):  # Me saco del medio el problemita de que el log de 0 da -infinito
    if SCH[i] == 0:
        SCH[i] = 0.0000001

SCHdB = 10*np.log10(SCH/np.max(SCH))  # Paso a dBFS


# Corto la señal original para comparar:
signal_edB = signal_edB[a:]
signal_edB = signal_edB[:b]

# Cálculo del tr:
# EDT:
x_last_MMv_EDT = np.where(TR_MMv < -10)[0][0]
x_last_SCH_EDT = np.where(SCHdB < -10)[0][0]

y_MMv_EDT = TR_MMv[0:x_last_MMv_EDT]
x_MMv_EDT = np.arange(0, x_last_MMv_EDT, 1).tolist()

b_MMv_EDT, m_MMv_EDT = cuadradosminimos(x_MMv_EDT, y_MMv_EDT)
EDT_MMv = (-60-b_MMv_EDT)/(m_MMv_EDT*fs)
print('El EDT por media movil es: {}'.format(EDT_MMv))

y_SCH_EDT = SCHdB[0:x_last_SCH_EDT]
x_SCH_EDT = np.arange(0, x_last_SCH_EDT, 1).tolist()

b_SCH_EDT, m_SCH_EDT = cuadradosminimos(x_SCH_EDT, y_SCH_EDT)
EDT_SCH = (-60-b_SCH_EDT)/(m_SCH_EDT*fs)
print('El EDT por Schroeder es: {}'.format(EDT_SCH))

# T10:
x_first_MMv_T10 = np.where(TR_MMv < -5)[0][0]
x_first_SCH_T10 = np.where(SCHdB < -5)[0][0]
x_last_MMv_T10 = np.where(TR_MMv < -15)[0][0]
x_last_SCH_T10 = np.where(SCHdB < -15)[0][0]

y_MMv_T10 = TR_MMv[x_first_MMv_T10:x_last_MMv_T10]
x_MMv_T10 = np.arange(x_first_MMv_T10, x_last_MMv_T10, 1).tolist()

b_MMv_T10, m_MMv_T10 = cuadradosminimos(x_MMv_T10, y_MMv_T10)
T10_MMv = (-60-b_MMv_T10)/(m_MMv_T10*fs)
print('El T10 por media movil es: {}'.format(T10_MMv))

y_SCH_T10 = SCHdB[x_first_SCH_T10:x_last_SCH_T10]
x_SCH_T10 = np.arange(x_first_SCH_T10, x_last_SCH_T10, 1).tolist()

b_SCH_T10, m_SCH_T10 = cuadradosminimos(x_SCH_T10, y_SCH_T10)
T10_SCH = (-60-b_SCH_T10)/(m_SCH_T10*fs)
print('El T10 por Schroeder es: {}'.format(T10_SCH))

# T20:
x_first_MMv_T20 = np.where(TR_MMv < -5)[0][0]
x_first_SCH_T20 = np.where(SCHdB < -5)[0][0]
x_last_MMv_T20 = np.where(TR_MMv < -25)[0][0]
x_last_SCH_T20 = np.where(SCHdB < -25)[0][0]

y_MMv_T20 = TR_MMv[x_first_MMv_T20:x_last_MMv_T20]
x_MMv_T20 = np.arange(x_first_MMv_T20, x_last_MMv_T20, 1).tolist()

b_MMv_T20, m_MMv_T20 = cuadradosminimos(x_MMv_T20, y_MMv_T20)
T20_MMv = (-60-b_MMv_T20)/(m_MMv_T20*fs)
print('El T20 por media movil es: {}'.format(T20_MMv))

y_SCH_T20 = SCHdB[x_first_SCH_T20:x_last_SCH_T20]
x_SCH_T20 = np.arange(x_first_SCH_T20, x_last_SCH_T20, 1).tolist()

b_SCH_T20, m_SCH_T20 = cuadradosminimos(x_SCH_T20, y_SCH_T20)
T20_SCH = (-60-b_SCH_T20)/(m_SCH_T20*fs)
print('El T20 por Schroeder es: {}'.format(T20_SCH))

# T30:
x_first_MMv_T30 = np.where(TR_MMv < -5)[0][0]
x_first_SCH_T30 = np.where(SCHdB < -5)[0][0]
x_last_MMv_T30 = np.where(TR_MMv < -35)[0][0]
x_last_SCH_T30 = np.where(SCHdB < -35)[0][0]

y_MMv_T30 = TR_MMv[x_first_MMv_T30:x_last_MMv_T30]
x_MMv_T30 = np.arange(x_first_MMv_T30, x_last_MMv_T30, 1).tolist()

b_MMv_T30, m_MMv_T30 = cuadradosminimos(x_MMv_T30, y_MMv_T30)
T30_MMv = (-60-b_MMv_T30)/(m_MMv_T30*fs)
print('El T30 por media movil es: {}'.format(T30_MMv))

y_SCH_T30 = SCHdB[x_first_SCH_T30:x_last_SCH_T30]
x_SCH_T30 = np.arange(x_first_SCH_T30, x_last_SCH_T30, 1).tolist()

b_SCH_T30, m_SCH_T30 = cuadradosminimos(x_SCH_T30, y_SCH_T30)
T30_SCH = (-60-b_SCH_T30)/(m_SCH_T30*fs)
print('El T30 por Schroeder es: {}'.format(T30_SCH))

EDT_fin_MMv, EDT_fin_SCH, T10_fin_MMv, T10_fin_SCH, T20_fin_MMv, T20_fin_SCH, T30_fin_MMv, T30_fin_SCH = TR(
    signal, fs, M)

EDT_fin_MMv.append(EDT_MMv)
EDT_fin_SCH.append(EDT_SCH)
T10_fin_MMv.append(T10_MMv)
T10_fin_SCH.append(T10_SCH)
T20_fin_MMv.append(T20_MMv)
T20_fin_SCH.append(T20_SCH)
T30_fin_MMv.append(T30_MMv)
T30_fin_SCH.append(T30_SCH)

centros = ['125 Hz', '250 Hz', '500 Hz', '1000 Hz', '2000 Hz', '4000 Hz', '8000 Hz', 'Global']

df = pd.DataFrame(data={'EDT media movil': EDT_fin_MMv, 'EDT Schroeder': EDT_fin_SCH, 'T10 media movil': T10_fin_MMv, 'T10 Schroeder': T10_fin_SCH,
                        'T20 media movil': T20_fin_MMv, 'T20 Schroeder': T20_fin_SCH, 'T30 media movil': T30_fin_MMv, 'T30 Schroeder': T30_fin_SCH},
                        index=centros)
print(df)

writer = pd.ExcelWriter('Datos_TR_codigo.xlsx', engine='xlsxwriter')
df.to_excel(writer, sheet_name='Hoja1', index=centros)
writer.save()

#Ploteo

plt.figure(1)

plt.plot(signal_edB, label='Envolvente de la RIR')
plt.plot(TR_MMv, label='Señal con filtro de media movil con M: ' + str(M))
plt.plot(SCHdB, label='Método de Schroeder')
plt.ylabel('Amplitud [dBFS]')
plt.xlabel('Muestras [n]')
plt.legend()
plt.show()
