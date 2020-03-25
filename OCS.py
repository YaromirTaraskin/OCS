from math import *
import numpy as np


def to_radians(fi):
    return pi * fi / 180

def dist(vector):
    return (vector.dot(vector)) ** 0.5


class ObliqueCoordinatesVector:
    def __init__(self, a, b, c):
        self.t = (a, b, c)

    def dot(self, other):
        """Скалярное произведение векторов"""
        sum = 0
        for i in range(3):
            for j in range(3):
                sum += self.t[i] * other.t[j] * g[i][j]
        return sum

    def minus(self, other):
        return ObliqueCoordinatesVector(self.t[0] - other.t[0], self.t[1] - other.t[1], self.t[2] - other.t[2])

    def plus(self, other):
        return ObliqueCoordinatesVector(self.t[0] + other.t[0], self.t[1] + other.t[1], self.t[2] + other.t[2])

    def to_cartesian(self):
        return np.array((
            a * self.t[0] + b * cos(gamma) * self.t[1] + c * cos(alpha) * cos(gamma) * self.t[2],
            b * sin(gamma) * self.t[1] + c * cos(alpha) * sin(gamma) * self.t[2],
            c * sin(alpha) * self.t[2]
        ))


def angle(v1, v2):
    """Угол между векторами"""
    return acos(v1.dot(v2) / (dist(v1) * dist(v2)))


def to_degrees(fi):
    return fi * 180 / pi


class Plane:
    """Плоскость"""
    def __init__(self, A, B, C):
        n = np.cross(B - A, C - A)
        self.kx = n[0]
        self.ky = n[1]
        self.kz = n[2]
        self.k = - n.dot(A)

    def normal(self):
        return np.array((self.kx, self.ky, self.kz))

    def angle(self, other):
        if type(other == Plane):
            return abs(angle(self.normal(), other.normal()))
        elif type(other == np.ndarray):
            return pi / 2 - angle(other, self.normal())

    def dist_to_point(self, A):
        return abs(kx * A[0] + ky * A[1] + kz * A[2] + k) / dist(self.normal())

'''
def angle(A, B, C):
    return angle(A - B, C - B)
'''
'''
A = ObliqueCoordinatesVector(0.38547, 0.26200, 0.98061)
B = ObliqueCoordinatesVector(0.30200, 0.26100, 0.92230)
AB = B.minus(A)
print(dist(AB))
C = ObliqueCoordinatesVector(0.32700, 0.33100, 1.03700)
AC = C.minus(A)
print(to_degrees(angle(AB, AC)))
cA = A.to_cartesian()
cB = B.to_cartesian()
cAB = cB - cA
cC = C.to_cartesian()
cAC = cC - cA
print(to_degrees(angle(cAB, cAC)))
print(dist(cAB))
'''
oblique_points = dict()
cartesian_points = dict()
oblique_vectors = dict()
cartesian_vectors = dict()
cartesian_planes = dict()

while True:
	n = input("""1 -- использовать параметры решётки(трансляции и углы между осями) для дигидрофосфата 2-метилбензимидазола
2 -- ввести параметры вручную
""")
	if n == 1:
		a = 6.84760
		b = 10.17780
		c = 11.08340
		alpha = 87.9530
		beta = 85.5460
		gamma = 74.7600
		break
	elif n == 2:
		a = input("Введите трансляцию a: ")
		b = input("Введите трансляцию b: ")
		c = input("Введите трансляцию c: ")
		alpha = input("Введите угол альфа(между осями c и b): ")
		beta = input("Введите угол бета(между осями a и c): ")
		gamma = input("Введите угол гамма(между осями a и b): ")
		break
		
alpha, beta, gamma = map(to_radians, (alpha, beta, gamma))
"""for i in a, b, c:
  print(i*i)
print(a*b*cos(gamma))
print(b*c*cos(alpha))
print(c*a*cos(beta))
"""
g = (
    (a * a, a * b * cos(gamma), a * c * cos(beta)),
    (a * b * cos(gamma), b * b, b * c * cos(alpha)),
    (a * c * cos(beta), b * c * cos(alpha), c*c)
)

while True:
    try:
        n = input("""1 -- добавить точку
2 -- добавить вектор
3 -- добавить плоскость
4 -- вывести точки
5 -- вывести вектора
6 -- вывести плоскости
7 -- найти расстояние (в ангстремах)
8 -- найти угол
exit -- выход
""")
        if n == '1':
            '''добавить точку'''
            name = input("Введите имя точки (если такая точка существует, она будет перезаписана): ")
            coordinates = tuple(map(float, input("Введите координаты через пробел: ").split()))
            oblique_points[name] = ObliqueCoordinatesVector(*coordinates)
            cartesian_points[name] = oblique_points[name].to_cartesian()
            print("Добавлено")
        elif n == '2':
            '''добавить вектор'''
            name = input("Введите имя вектора: ")
            start = input("Введите имя точки начала: ")
            end = input("Введите имя точки конца: ")
            oblique_vectors[name] = oblique_points[end].minus(oblique_points[start])
            cartesian_vectors[name] = oblique_vectors[name].to_cartesian()
            print("Добавлено")
        elif n == '3':
            '''добавить плоскость'''
            name = input("Введите имя плоскости: ")
            points_names = input("Ведите мена 3х точек: ").split()
            points = [cartesian_points[point] for point in points_names]
            cartesian_planes[name] = Plane(*points)
            print("Добавлено")
        elif n == '4':
            '''вывести точки'''
            k = input("""1 -- вывести точки с координатами\n2 -- вывести точки без координат\n""")
            if k == '1':
                for name, coordinates in oblique_points.items():
                    print(name, ": (", ", ".join(map(str, coordinates.t)), ")", sep='')
            elif k == '2':
                print(", ".join(oblique_points.keys()))
        elif n == '5':
            '''вывести вектора'''
            k = input("""1 -- вывести вектора с координатами\n2 -- вывести вектора без координат\n""")
            if k == '1':
                '''вывести вектора с координатами'''
                for name, coordinates in oblique_vectors.items():
                    print(name, ": (", ", ".join(map(str, coordinates.t)), ")", sep='')
            elif k == '2':
                '''вывести вектора без координат'''
                print(", ".join(oblique_vectors.keys()))
        elif n == '6':
            '''вывести плоскости'''
            print(", ".join(cartesian_planes.keys()))
        elif n == '7':
            '''найти расстояние'''
            k = input("1 -- длина вектора\n2 -- расстояние между точками\n3 -- расстояние от точки до плоскости\n")
            if k == '1':
                '''длина вектора'''
                vec = oblique_vectors[input("Введите имя вектора: ")]
                print(dist(vec))
            elif k == '2':
                '''расстояние между точками'''
                point1 = oblique_points[input("Введите имя первой точки: ")]
                point2 = oblique_points[input("Введите имя второй точки: ")]
                vec = point2.minus(point1)
                print(dist(vec))
            elif k == '3':
                '''расстояние от точки до плоскости'''
                point = cartesian_points[input("Введите имя точки: ")]
                plane = cartesian_planes[input("Введите имя плоскости: ")]
                print(plane.dist_to_point(point))
        elif n == '8':
            '''найти угол'''
            k = input("""1 -- угол между векторами\n2 -- угол между плоскостями\n3 -- угол между плоскостью и вектором\n4 -- угол по трём точкам""")
            if k == '1':
                '''угол между векторами'''
                vec1 = oblique_vectors[input("Введите имя первого вектора: ")]
                vec2 = oblique_vectors[input("Введите имя второго вектора: ")]
                print(angle(vec1, vec2))
            elif k == '2':
                '''угол между плоскостями'''
                plane1 = cartesian_planes[input("Введите имя первой плоскости: ")]
                plane2 = cartesian_planes[input("Введите имя второй плоскости: ")]
                print(plane1.angle(plane2))
            elif k == '3':
                '''угол между плоскостью и вектором'''
                plane = cartesian_planes[input("Введите имя плоскости: ")]
                vec = cartesian_vectors[input("Введите имя вектора: ")]
                print(plane.angle(vec))
            elif k == '4':
                '''угол по трём точкам'''
                A = oblique_points[input("Введите имя первой точки: ")]
                B = oblique_points[input("Введите имя второй точки: ")]
                C = oblique_points[input("Введите имя третьей точки: ")]
                BA = A.minus(B)
                BC = C.minus(B)
                print(angle(BC, BA))
        elif n == 'exit':
            exit(0)
        else:
            print("Неверный ввод")
        input("Нажмите Enter, чтобы вернуться в главное меню...")
    except Exception:
        print("Ошибка")
        raise

