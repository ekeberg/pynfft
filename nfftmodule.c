#include <Python.h>
#include "structmember.h"
#include <numpy/arrayobject.h>
#include <nfft3.h>
#include <math.h>
#include <stdio.h>

//#include "nfftclassmodule.h"

PyDoc_STRVAR(nfft__doc__, "nfft(real_space, coordinates)\n\nCalculate nfft from arbitrary dimensional array.\n\real_space should be an array (or any object that can trivially be converted to one.\ncoordinates should be a NxD array where N is the number of points where the Fourier transform should be evaluated and D is the dimensionality of the input array\ndirect (optional) requires the use of the more accurate but slower ndft (default is False)");
static PyObject *nfft(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyObject *in_obj, *coord_obj;
  PyObject *use_direct_obj = NULL;
  
  /*
  if (!PyArg_ParseTuple(args, "OO", &in_obj, &coord_obj)) {
    return NULL;
  }
  */
  static char *kwlist[] = {"real_space", "coordinates", "use_direct", NULL};
  //static char *kwlist[] = {"direct", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|O", kwlist, &in_obj, &coord_obj, &use_direct_obj)) {
  //if (!PyArg_ParseTuple(args, "OO|O", &in_obj, &coord_obj, &use_direct_obj)) {
    return NULL;
  }
  int use_direct = 0;
  if (use_direct_obj != NULL && PyObject_IsTrue(use_direct_obj)) {
    use_direct = 1;
  }

  PyObject *coord_array = PyArray_FROM_OTF(coord_obj, NPY_DOUBLE, NPY_IN_ARRAY);
  PyObject *in_array = PyArray_FROM_OTF(in_obj, NPY_COMPLEX128, NPY_IN_ARRAY);
  if (coord_array == NULL || in_array == NULL) {
    Py_XDECREF(coord_array);
    Py_XDECREF(in_array);
    return NULL;
  }

  int ndim = PyArray_NDIM(in_array);
  if (ndim <= 0) {
    PyErr_SetString(PyExc_ValueError, "Input array can't be 0 dimensional\n");
    return NULL;
  }

  if ((PyArray_NDIM(coord_array) != 2 || PyArray_DIM(coord_array, 1) != ndim) && (ndim != 1 || PyArray_NDIM(coord_array) != 1)) {
    PyErr_SetString(PyExc_ValueError, "Coordinates must be given as array of dimensions [NUMBER_OF_POINTS, NUMBER_OF_DIMENSIONS] of [NUMBER_OF_POINTS for 1D transforms.\n");
    Py_XDECREF(coord_array);
    Py_XDECREF(in_array);
    return NULL;
  }
  int number_of_points = (int) PyArray_DIM(coord_array, 0);

  nfft_plan my_plan;
  if (ndim == 1) {
    int dim_x = (int)PyArray_DIM(in_array, 0);
    nfft_init_1d(&my_plan, dim_x, number_of_points);
    memcpy(my_plan.f_hat, PyArray_DATA(in_array), dim_x*sizeof(fftw_complex));
  } else if (ndim == 2) {
    int dim_y = (int)PyArray_DIM(in_array, 0);
    int dim_x = (int)PyArray_DIM(in_array, 1);
    nfft_init_2d(&my_plan, dim_y, dim_x, number_of_points);
    memcpy(my_plan.f_hat, PyArray_DATA(in_array), dim_y*dim_x*sizeof(fftw_complex));
  } else if (ndim == 3) {
    int dim_z = (int)PyArray_DIM(in_array, 0);
    int dim_y = (int)PyArray_DIM(in_array, 1);
    int dim_x = (int)PyArray_DIM(in_array, 2);
    nfft_init_3d(&my_plan, dim_z, dim_y, dim_x, number_of_points);
    memcpy(my_plan.f_hat, PyArray_DATA(in_array), dim_z*dim_y*dim_x*sizeof(fftw_complex));
  } else {
    int total_number_of_pixels = 1;
    int dims[ndim];
    for (int dim = 0; dim < ndim; ++dim) {
      dims[dim] = (int)PyArray_DIM(in_array, dim);
      total_number_of_pixels *= dims[dim];
    }
    nfft_init(&my_plan, ndim, dims, number_of_points);
    memcpy(my_plan.f_hat, PyArray_DATA(in_array), total_number_of_pixels*sizeof(fftw_complex));
  }
  memcpy(my_plan.x, PyArray_DATA(coord_array), ndim*number_of_points*sizeof(double));
  
  if (my_plan.nfft_flags &PRE_PSI) {
    nfft_precompute_one_psi(&my_plan);
  }

  if (use_direct == 1) {
    nfft_trafo_direct(&my_plan);
  } else {
    nfft_trafo(&my_plan);
  }

  //npy_intp out_dim[] = {number_of_points};
  int out_dim[] = {number_of_points};
  //PyObject *vecout_array = PyArray_FromDims(1, vecout_dims, NPY_COMPLEX128);
  //PyObject *out_array = (PyObject *)PyArray_SimpleNew(1, out_dim, NPY_COMPLEX128);
  PyObject *out_array = (PyObject *)PyArray_FromDims(1, out_dim, NPY_COMPLEX128);
  memcpy(PyArray_DATA(out_array), my_plan.f, number_of_points*sizeof(fftw_complex));

  nfft_finalize(&my_plan);
  return out_array;
}

PyDoc_STRVAR(nfft_inplace__doc__, "nfft_inplace(real_space, coordinates)\n\nCalculate nfft from arbitrary dimensional array.\n\real_space should be an array (or any object that can trivially be converted to one.\ncoordinates should be a NxD array where N is the number of points where the Fourier transform should be evaluated and D is the dimensionality of the input array\noutput_array should be ndarray of type complex128. The is written to here, if the array is a continuous block in memory this can speed up the calculation.\ndirect (optional) requires the use of the more accurate but slower ndft (default is False).");
static PyObject *nfft_inplace(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyObject *in_obj, *coord_obj, *out_obj;
  PyObject *use_direct_obj = NULL;

  static char *kwlist[] = {"real_space", "coordinates", "output", "use_direct", NULL};
  //if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|O", kwlist, &in_obj, &coord_obj, &use_direct_obj)) {
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOO|O", kwlist, &in_obj, &coord_obj, &out_obj, &use_direct_obj)) {
    return NULL;
  }
  int use_direct = 0;
  if (use_direct_obj != NULL && PyObject_IsTrue(use_direct_obj)) {
    use_direct = 1;
  }

  PyObject *coord_array = PyArray_FROM_OTF(coord_obj, NPY_DOUBLE, NPY_IN_ARRAY);
  PyObject *in_array = PyArray_FROM_OTF(in_obj, NPY_COMPLEX128, NPY_IN_ARRAY);
  if (coord_array == NULL || in_array == NULL) {
    Py_XDECREF(coord_array);
    Py_XDECREF(in_array);
    return NULL;
  }

  int ndim = PyArray_NDIM(in_array);
  if (ndim <= 0) {
    PyErr_SetString(PyExc_ValueError, "Input array can't be 0 dimensional\n");
    return NULL;
  }

  if ((PyArray_NDIM(coord_array) != 2 || PyArray_DIM(coord_array, 1) != ndim) && (ndim != 1 || PyArray_NDIM(coord_array) != 1)) {
    PyErr_SetString(PyExc_ValueError, "Coordinates must be given as array of dimensions [NUMBER_OF_POINTS, NUMBER_OF_DIMENSIONS] of [NUMBER_OF_POINTS for 1D transforms.\n");
    Py_XDECREF(coord_array);
    Py_XDECREF(in_array);
    return NULL;
  }
  int number_of_points = (int) PyArray_DIM(coord_array, 0);

  if (!PyArray_Check(out_obj)) {
    PyErr_SetString(PyExc_ValueError, "Output must be numpy.array of dtype complex128");
    Py_XDECREF(coord_array);
    Py_XDECREF(in_array);
    return NULL;
  }

  if (PyArray_DESCR(out_obj)->type_num != NPY_COMPLEX128) {
    PyErr_SetString(PyExc_ValueError, "Output array must be of dtype complex128");
    Py_XDECREF(coord_array);
    Py_XDECREF(in_array);
    return NULL;
  }

  PyObject *out_array = PyArray_FROM_OTF(out_obj, NPY_COMPLEX128, NPY_INOUT_ARRAY);
  if (PyArray_NDIM(out_array) != 1 || PyArray_DIM(out_array, 0) != number_of_points) {
    PyErr_SetString(PyExc_ValueError,  "Output array must be one dimensional and same length as coordinates array.");
    Py_XDECREF(coord_array);
    Py_XDECREF(in_array);
    Py_XDECREF(out_array);
    return NULL;
  }

  nfft_plan my_plan;
  if (ndim == 1) {
    int dim_x = (int)PyArray_DIM(in_array, 0);
    nfft_init_1d(&my_plan, dim_x, number_of_points);
    memcpy(my_plan.f_hat, PyArray_DATA(in_array), dim_x*sizeof(fftw_complex));
  } else if (ndim == 2) {
    int dim_y = (int)PyArray_DIM(in_array, 0);
    int dim_x = (int)PyArray_DIM(in_array, 1);
    nfft_init_2d(&my_plan, dim_y, dim_x, number_of_points);
    memcpy(my_plan.f_hat, PyArray_DATA(in_array), dim_y*dim_x*sizeof(fftw_complex));
  } else if (ndim == 3) {
    int dim_z = (int)PyArray_DIM(in_array, 0);
    int dim_y = (int)PyArray_DIM(in_array, 1);
    int dim_x = (int)PyArray_DIM(in_array, 2);
    nfft_init_3d(&my_plan, dim_z, dim_y, dim_x, number_of_points);
    memcpy(my_plan.f_hat, PyArray_DATA(in_array), dim_z*dim_y*dim_x*sizeof(fftw_complex));
  } else {
    int total_number_of_pixels = 1;
    int dims[ndim];
    for (int dim = 0; dim < ndim; ++dim) {
      dims[dim] = (int)PyArray_DIM(in_array, dim);
      total_number_of_pixels *= dims[dim];
    }
    nfft_init(&my_plan, ndim, dims, number_of_points);
    memcpy(my_plan.f_hat, PyArray_DATA(in_array), total_number_of_pixels*sizeof(fftw_complex));
  } 

  memcpy(my_plan.x, PyArray_DATA(coord_array), ndim*number_of_points*sizeof(double));
  
  if (my_plan.nfft_flags &PRE_PSI) {
    nfft_precompute_one_psi(&my_plan);
  }

  if (use_direct == 1) {
    nfft_trafo_direct(&my_plan);
  } else {
    nfft_trafo(&my_plan);
  }
  memcpy(PyArray_DATA(out_array), my_plan.f, number_of_points*sizeof(fftw_complex));

  nfft_finalize(&my_plan);
  return Py_BuildValue("i", 1);
}


/*
static PyObject *nfft3(PyObject *self, PyObject *args)
{
  PyObject *vecin_obj, *veccoord_obj;
  double *cin, *cout, *ccoord;

  int number_of_pixels_x, number_of_pixels_y, number_of_pixels_z;
  int number_of_points;
  int i;

  if (!PyArg_ParseTuple(args, "OO", &vecin_obj, &veccoord_obj))
    return NULL;
  PyObject *veccoord_array = PyArray_FROM_OTF(veccoord_obj, NPY_DOUBLE, NPY_IN_ARRAY);
  PyObject *vecin_array = PyArray_FROM_OTF(vecin_obj, NPY_COMPLEXLTR, NPY_IN_ARRAY);
  if (veccoord_array == NULL || vecin_array == NULL) {
    Py_XDECREF(veccoord_array);
    Py_XDECREF(vecin_array);
    return NULL;
  }

  if (PyArray_NDIM(vecin_array) != 3) {
    Py_XDECREF(veccoord_array);
    Py_XDECREF(vecin_array);
    PyErr_SetString(PyExc_ValueError, "Input array must be three dimensional");
    return NULL;
  }
  number_of_pixels_z = (int)PyArray_DIM(vecin_array, 0);
  number_of_pixels_y = (int)PyArray_DIM(vecin_array, 1);
  number_of_pixels_x = (int)PyArray_DIM(vecin_array, 2);

  if (PyArray_NDIM(veccoord_array) != 2 || PyArray_DIM(veccoord_array, 1) != 3) {
    Py_XDECREF(veccoord_array);
    Py_XDECREF(vecin_array);
    PyErr_SetString(PyExc_ValueError, "Coordinate array must be three dimensional");
    return NULL;
  }    
  number_of_points = (int)PyArray_DIM(veccoord_array, 0);

  ccoord = PyArray_DATA(veccoord_array);
  cin = PyArray_DATA(vecin_array);

  int vecout_dims[] = {number_of_points};
  PyObject *vecout_array = PyArray_FromDims(1, vecout_dims, NPY_COMPLEX128);
  cout = PyArray_DATA(vecout_array);



  //return Py_BuildValue("i", 1);
  Py_XDECREF(veccoord_array);
  Py_XDECREF(vecin_array);
  return vecout_array;
}
*/

PyDoc_STRVAR(Transformer__doc__, "Transformer(real_space)\n\nCreates an object that can be used to calculate multiple nfft transforms from the same array.");
typedef struct {
  PyObject_HEAD
  nfft_plan my_plan;
  PyArrayObject *real_map;
  fftw_complex *sneaky_ref;
  int ndim;
  int max_number_of_points;
}Transformer;

static PyMemberDef Transformer_members[] = {
  {"real_map", T_OBJECT_EX, offsetof(Transformer, real_map), 0, "Real space map."},
  {NULL}
};

static int Transformer_init(Transformer *self, PyObject *args, PyObject *kwds)
{
  PyObject *input_obj;
  if (!PyArg_ParseTuple(args, "Oi", &input_obj, &self->max_number_of_points)) {
    return -1;
  }
  PyObject *input_array = PyArray_FROM_OTF(input_obj, NPY_COMPLEX128, NPY_IN_ARRAY);
  if (input_array == NULL) {
    return -1;
  }
  self->real_map = (PyArrayObject *)input_array;

  self->ndim = PyArray_NDIM(input_array);
  int total_number_of_pixels = 1;
  int dims[self->ndim];
  for (int dim = 0; dim < self->ndim; ++dim) {
    dims[dim] = (int) PyArray_DIM(input_array, dim);
    total_number_of_pixels *= dims[dim];
  }
  nfft_init(&self->my_plan, self->ndim, dims, self->max_number_of_points);
  //free(self->my_plan.f_hat);
  self->sneaky_ref = self->my_plan.f_hat;
  self->my_plan.f_hat = (fftw_complex *)PyArray_DATA(self->real_map);
  return 0;
}

static void Transformer_dealloc(Transformer *self)
{
  Py_XDECREF(self->real_map);
  self->my_plan.f_hat = self->sneaky_ref;
  nfft_finalize(&self->my_plan);
  self->ob_type->tp_free((PyObject *)self);
}

PyDoc_STRVAR(Transformer_transform__doc__, "transform(coordinates)\n\nReturns transformation at given coordintase.\n\nCoordinate array has format [NUMBER_OF_POINTS, NUMBER_OF_DIMENSIONS] and can be of any type that can direcly be converted to an ndarray.");
static PyObject *Transformer_transform(Transformer *self, PyObject *args, PyObject *kwargs)
{
  PyObject *input_obj;
  PyObject *use_direct_obj = NULL;
  //if (!PyArg_ParseTuple(args, "O", &input_obj)) {
  static char *kwlist[] = {"coordinates", "use_direct", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|O", kwlist, &input_obj, &use_direct_obj)) {
    return NULL;
  }
  PyObject *coordinates_array = PyArray_FROM_OTF(input_obj, NPY_FLOAT64, NPY_IN_ARRAY);
  
  if (coordinates_array == NULL) {
    return NULL;
  }

  if ((PyArray_NDIM(coordinates_array) != 2 || PyArray_DIM(coordinates_array, 1) != self->ndim) && (self->ndim != 1 || PyArray_NDIM(coordinates_array) != 1)) {
    PyErr_SetString(PyExc_ValueError, "Coordinates must be given as array of dimensions [NUMBER_OF_POINTS, NUMBER_OF_DIMENSIONS] or [NUMBER_OF_POINTS for 1D transforms.\n");
    Py_XDECREF(coordinates_array);
    return NULL;
  }

  int number_of_points = (int) PyArray_DIM(coordinates_array, 0);
  if (number_of_points > self->max_number_of_points) {
    PyErr_SetString(PyExc_ValueError, "Coordinates array is longer than max_number_of_coordinates");
    return NULL;
  }

  int use_direct = 0;
  if (use_direct_obj != NULL && PyObject_IsTrue(use_direct_obj)) {
    use_direct = 1;
  }

  memcpy(self->my_plan.x, PyArray_DATA(coordinates_array), self->ndim*number_of_points*sizeof(double));

  if (self->my_plan.nfft_flags &PRE_PSI) {
    nfft_precompute_one_psi(&self->my_plan);
  }

  if (use_direct == 1) {
    nfft_trafo_direct(&self->my_plan);
  } else {
    nfft_trafo(&self->my_plan);
  }

  npy_intp out_dim[] = {number_of_points};
  PyObject *out_array = (PyObject *)PyArray_SimpleNew(1, out_dim, NPY_COMPLEX128);
  memcpy(PyArray_DATA(out_array), self->my_plan.f, number_of_points*sizeof(fftw_complex));
  return out_array;
  /*
  nfft_plan my_plan;
  nfft_init_3d(&my_plan, number_of_pixels_z, number_of_pixels_y, number_of_pixels_x, number_of_points);

  memcpy(my_plan.x, coord, 3*number_of_points*sizeof(double));
  memcpy(my_plan.f_hat, in, 2*number_of_pixels_x*number_of_pixels_y*number_of_pixels_z*sizeof(double));

  if (my_plan.nfft_flags &PRE_PSI) {
    nfft_precompute_one_psi(&my_plan);
  }
  
  nfft_trafo(&my_plan);
  memcpy(out, my_plan.f, 2*number_of_points*sizeof(double));

  nfft_finalize(&my_plan);
  */

  return Py_BuildValue("");
}

PyDoc_STRVAR(Transformer_ndim__doc__, "ndim()\n\nGet the number of dimensions.");
static PyObject *Transformer_ndim(Transformer *self, PyObject *args, PyObject *kwds)
{
  if (self->ndim > 0) {
    return Py_BuildValue("i", self->ndim);
  } else {
    return Py_BuildValue("");
  }
}

PyDoc_STRVAR(Transformer_max_number_of_points__doc__, "max_number_of_points()\n\nGet the maximum length of coordinates array.");
static PyObject *Transformer_max_number_of_points(Transformer *self, PyObject *args, PyObject *kwds)
{
  return Py_BuildValue("i", self->max_number_of_points);
}

static PyMethodDef Transformer_methods[] = {
  {"transform", (PyCFunction) Transformer_transform, METH_VARARGS|METH_KEYWORDS, Transformer_transform__doc__},
  {"ndim", (PyCFunction) Transformer_ndim, METH_VARARGS, Transformer_ndim__doc__},
  {"max_number_of_points", (PyCFunction) Transformer_max_number_of_points, METH_VARARGS, Transformer_max_number_of_points__doc__},
  {NULL}
};

static PyTypeObject TransformerType = {
   PyObject_HEAD_INIT(NULL)
   0,                         /* ob_size */
   "Transformer",         /* tp_name */
   sizeof(Transformer),   /* tp_basicsize */
   0,                         /* tp_itemsize */
   (destructor)Transformer_dealloc, /* tp_dealloc */
   0,                         /* tp_print */
   0,                         /* tp_getattr */
   0,                         /* tp_setattr */
   0,                         /* tp_compare */
   0,                         /* tp_repr */
   0,                         /* tp_as_number */
   0,                         /* tp_as_sequence */
   0,                         /* tp_as_mapping */
   0,                         /* tp_hash */
   0,                         /* tp_call */
   0,                         /* tp_str */
   0,                         /* tp_getattro */
   0,                         /* tp_setattro */
   0,                         /* tp_as_buffer */
   Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags*/
   Transformer__doc__,        /* tp_doc */
   0,                         /* tp_traverse */
   0,                         /* tp_clear */
   0,                         /* tp_richcompare */
   0,                         /* tp_weaklistoffset */
   0,                         /* tp_iter */
   0,                         /* tp_iternext */
   Transformer_methods,   /* tp_methods */
   Transformer_members,   /* tp_members */
   0,                         /* tp_getset */
   0,                         /* tp_base */
   0,                         /* tp_dict */
   0,                         /* tp_descr_get */
   0,                         /* tp_descr_set */
   0,                         /* tp_dictoffset */
   (initproc)Transformer_init,  /* tp_init */
   0,                         /* tp_alloc */
   0,                         /* tp_new */
};

static PyMethodDef NfftMethods[] = {
  {"nfft", (PyCFunction)nfft, METH_VARARGS|METH_KEYWORDS, nfft__doc__},
  {"nfft_inplace", (PyCFunction)nfft_inplace , METH_VARARGS|METH_KEYWORDS, nfft_inplace__doc__},
  {NULL, NULL, 0, NULL}
};



PyMODINIT_FUNC initnfft(void)
{
  import_array();
  PyObject *m = Py_InitModule3("nfft", NfftMethods, "Nonequispaced FFT tools.");
  if (m == NULL)
    return;

  TransformerType.tp_new = PyType_GenericNew;
  if (PyType_Ready(&TransformerType) < 0)
    return;

  Py_INCREF(&TransformerType);
  PyModule_AddObject(m, "Transformer", (PyObject *)&TransformerType);
    
}
/*
int main(int argc, char *argv[])
{
  Py_SetProgramName(argv[0]);
  Py_Initialize();
  initnfft_c();
  return 0;
}
*/
