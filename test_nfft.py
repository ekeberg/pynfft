import numpy
import nfft
import unittest

class TestNfft(unittest.TestCase):
    def setUp(self):
        self._size = 10
        self._decimals = 10
        self._coord_1d = numpy.linspace(-0.5, 0.5-1./self._size, self._size)
        self._2d_coord = [[3, 5], [8, 7]]
        self._3d_coord = [[3, 5, 2], [8, 7, 4]]

    def test_transformer_1d(self):
        a = numpy.random.random(self._size)
        t = nfft.Transformer(a)
        ft_nfft = t.transform(self._coord_1d)
        ft_fftw = numpy.fft.fftshift(numpy.fft.fft(numpy.fft.fftshift(a)))
        numpy.testing.assert_almost_equal(ft_nfft, ft_fftw, decimal=self._decimals)

    def test_transformer_2d(self):
        a = numpy.random.random((self._size, )*2)
        t = nfft.Transformer(a)
        ft_nfft = t.transform([[self._coord_1d[self._2d_coord[0][0]], self._coord_1d[self._2d_coord[0][1]]],
                               [self._coord_1d[self._2d_coord[1][0]], self._coord_1d[self._2d_coord[1][1]]]])
        ft_fftw = numpy.fft.fftshift(numpy.fft.fft2(numpy.fft.fftshift(a)))
        numpy.testing.assert_almost_equal(ft_nfft, ft_fftw[(self._2d_coord[0][0], self._2d_coord[1][0]), (self._2d_coord[0][1], self._2d_coord[1][1])])

    def test_nfft(self):
        a = numpy.random.random(self._size)
        ft_nfft = nfft.nfft(a, self._coord_1d)
        ft_fftw = numpy.fft.fftshift(numpy.fft.fft(numpy.fft.fftshift(a)))
        numpy.testing.assert_almost_equal(ft_nfft, ft_fftw, decimal=self._decimals)

    def test_nfft_inplace(self):
        a = numpy.random.random(self._size)
        ft_nfft = numpy.empty(self._size, dtype="complex128")
        nfft.nfft_inplace(a, self._coord_1d, ft_nfft)
        ft_fftw = numpy.fft.fftshift(numpy.fft.fft(numpy.fft.fftshift(a)))
        numpy.testing.assert_almost_equal(ft_nfft, ft_fftw, decimal=self._decimals)

    def test_nfft_inplace_subarray(self):
        a = numpy.random.random(self._size)
        ft_nfft_large = numpy.zeros(self._size*2, dtype="complex128")
        ft_nfft = ft_nfft_large[::2]
        nfft.nfft_inplace(a, self._coord_1d, ft_nfft)
        ft_fftw = numpy.fft.fftshift(numpy.fft.fft(numpy.fft.fftshift(a)))
        numpy.testing.assert_almost_equal(ft_nfft, ft_fftw, decimal=self._decimals)

    def test_nfft3(self):
        a = numpy.random.random((self._size, )*3)
        ft_nfft = nfft.nfft3(a, [[self._coord_1d[self._3d_coord[0][0]], self._coord_1d[self._3d_coord[0][1]], self._coord_1d[self._3d_coord[0][2]]],
                                 [self._coord_1d[self._3d_coord[1][0]], self._coord_1d[self._3d_coord[1][1]], self._coord_1d[self._3d_coord[1][2]]]])
        ft_fftw = numpy.fft.fftshift(numpy.fft.fftn(numpy.fft.fftshift(a)))
        numpy.testing.assert_almost_equal(ft_nfft, ft_fftw[(self._3d_coord[0][0], self._3d_coord[1][0]),
                                                           (self._3d_coord[0][1], self._3d_coord[1][1]),
                                                           (self._3d_coord[0][2], self._3d_coord[1][2])])
    def test_nfft3_inplace(self):
        a = numpy.random.random((self._size, )*3)
        ft_nfft = numpy.empty(2, dtype="complex128")
        nfft.nfft3_inplace(a, [[self._coord_1d[self._3d_coord[0][0]], self._coord_1d[self._3d_coord[0][1]], self._coord_1d[self._3d_coord[0][2]]],
                               [self._coord_1d[self._3d_coord[1][0]], self._coord_1d[self._3d_coord[1][1]], self._coord_1d[self._3d_coord[1][2]]]], ft_nfft)
        ft_fftw = numpy.fft.fftshift(numpy.fft.fftn(numpy.fft.fftshift(a)))
        numpy.testing.assert_almost_equal(ft_nfft, ft_fftw[(self._3d_coord[0][0], self._3d_coord[1][0]),
                                                           (self._3d_coord[0][1], self._3d_coord[1][1]),
                                                           (self._3d_coord[0][2], self._3d_coord[1][2])])
        
    def test_nfftn_1d(self):
        a = numpy.random.random(self._size)
        ft_nfft = nfft.nfftn(a, self._coord_1d)
        ft_fftw = numpy.fft.fftshift(numpy.fft.fft(numpy.fft.fftshift(a)))
        numpy.testing.assert_almost_equal(ft_nfft, ft_fftw, decimal=self._decimals)

    def test_nfftn_inplace_1d(self):
        a = numpy.random.random(self._size)
        ft_nfft = numpy.empty(self._size, dtype="complex128")
        nfft.nfftn_inplace(a, self._coord_1d, ft_nfft)
        ft_fftw = numpy.fft.fftshift(numpy.fft.fft(numpy.fft.fftshift(a)))
        numpy.testing.assert_almost_equal(ft_nfft, ft_fftw, decimal=self._decimals)
            
    def test_nfftn_2d(self):
        pass

    def test_nfftn_inplace_2d(self):
        pass

    def test_failures(self):
        a = numpy.random.random(self._size)
        self.assertRaises(TypeError, nfft.nfft, (a, "hej"))
        self.assertRaises(TypeError, nfft.nfft, ("hej", a))

if __name__ == "__main__":
    unittest.main()

# size = 10
# coord_1d = numpy.linspace(-0.5, 0.5-1./size, size)

# def test_transformer():
#     a = numpy.random.random(size)
#     t = nfft.Transformer(a)
#     print t.transform(coord_1d)
#     print numpy.fft.fftshift(numpy.fft.fft(numpy.fft.fftshift(a)))

#     a = numpy.random.random((size, size))
#     t = nfft.Transformer(a)
#     print t.transform([(coord_1d[0], coord_1d[0]), (coord_1d[1], coord_1d[0])])
#     fftw_out = numpy.fft.fftshift(numpy.fft.fft2(numpy.fft.fftshift(a)))
#     print numpy.array([fftw_out[0, 0], fftw_out[1, 0]])

# saved_value = t.real_map[3, 4]
# t.real_map[3, 4] = -3.+1.j
# print t.transform([(coord_1d[0], coord_1d[0]), (coord_1d[1], coord_1d[0])])
# t.real_map[3, 4] = saved_value
# print t.transform([(coord_1d[0], coord_1d[0]), (coord_1d[1], coord_1d[0])])



#a = numpy.random.random((10, 10, 10))
#print nfft.nfft3(a, [[0.3, 0.3, 0.3], [0.2, 0.0, 0.1]])

# size = 10
# #a = numpy.random.random(size)
# #coordinates = [0.3, 0.2]
# #coordinates = numpy.linspace(-0.5+1./size/2., 0.5-1./size/2., size)
# #coordinates = numpy.linspace(-0.5, 0.5, size)
# #coordinates = numpy.linspace(-0.5, 0.5-1./size, size)

# #out1 = nfft.nfft(a, coordinates)
# #print out1
# #print numpy.fft.fftshift(numpy.fft.fft(numpy.fft.fftshift(a)))

# # out2 = numpy.empty(2, dtype="complex128")
# # nfft.nfft_inplace(a, coordinates, out2)
# # print out2

# b = numpy.random.random((10, 10, 10))
# #coordinates = [[0.3, 0.3, 0.3], [0.2, 0.0, 0.1]]
# #coordinates = [[0., 0., 0.3], [0., 0., 0.2]]
# coord_1d = numpy.linspace(-0.5, 0.5-1./size, size)
# coordinates = [(coord_1d[0], coord_1d[0], coord_1d[0]),
#                (coord_1d[0], coord_1d[0], coord_1d[1])]

# out1 = nfft.nfft3(b, coordinates)
# print out1
# out2 = numpy.fft.fftshift(numpy.fft.fftn(numpy.fft.fftshift(b)))
# print [out2[0, 0, 0], out2[0, 0, 1]]

# # out2 = numpy.empty(2, dtype="complex128")
# # nfft.nfft3_inplace(b, coordinates, out2)
# # print out2
