#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <stdbool.h>
#include <complex.h>
#include <math.h>

PyObject * get_none_value() {
    Py_INCREF(Py_None);
    return Py_None;
}

// IntArray ===================================================================
typedef struct {
    PyObject_HEAD
    size_t * elements;
    size_t count;
} IntArray;

static PyObject * IntArray_new(PyTypeObject* type, PyObject * args, PyObject kw)
{
    IntArray * self = (IntArray *) type->tp_alloc(type, 0);
    if (self != NULL) {
        self->elements = NULL;
        self->count = 0;
    }
    return (PyObject *) self;
}

static void IntArray_dealloc(IntArray * self)
{
    free(self->elements);
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static PyObject* IntArray_getitem(IntArray * self, Py_ssize_t index) {
    if (index < 0 || *((size_t*) &index) >= self->count) {
        PyErr_Format(PyExc_IndexError, "IntArray[%zd] called on array with %zu items",
            index, self->count);
        return NULL;
    }
    return PyLong_FromSize_t(self->elements[index]);
}

static Py_ssize_t IntArray_len(IntArray * self) {
    return self->count;
}

static PySequenceMethods IntArrayTypeAsSeq = {
    .sq_item = (ssizeargfunc) IntArray_getitem,
    .sq_length = (lenfunc) IntArray_len,
};

static PyTypeObject IntArrayType = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "mandelbrot_native.IntArray",
    .tp_basicsize = sizeof(IntArray),
    .tp_new = (newfunc) IntArray_new,
    .tp_dealloc = (destructor) IntArray_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_as_sequence = &IntArrayTypeAsSeq
};


static IntArray * create_int_array(size_t count) {
    IntArray * array = (IntArray *) PyObject_CallObject((PyObject *) &IntArrayType, NULL);
    if (array != NULL) {
        array->count = count;
        array->elements = malloc(sizeof(size_t) * count);
    }
    return array;
}

// FloatArray ===================================================================
typedef struct {
    PyObject_HEAD
    double * elements;
    size_t count;
} FloatArray;

static PyObject * FloatArray_new(PyTypeObject* type, PyObject * args, PyObject kw)
{
    FloatArray * self = (FloatArray *) type->tp_alloc(type, 0);
    if (self != NULL) {
        self->elements = NULL;
        self->count = 0;
    }
    return (PyObject *) self;
}

static void FloatArray_dealloc(FloatArray * self)
{
    free(self->elements);
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static PyObject * FloatArray_getitem(FloatArray * self, Py_ssize_t index) {
    if (index < 0 || *((size_t*) &index) >= self->count) {
        PyErr_Format(PyExc_IndexError, "FloatArray[%zd] called on array with %zu items",
            index, self->count);
        return NULL;
    }
    return PyFloat_FromDouble(self->elements[index]);
}

static Py_ssize_t FloatArray_len(FloatArray * self) {
    return self->count;
}

static PySequenceMethods FloatArrayTypeAsSeq = {
    .sq_item = (ssizeargfunc) FloatArray_getitem,
    .sq_length = (lenfunc) FloatArray_len,
};

static PyTypeObject FloatArrayType = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "mandelbrot_native.FloatArray",
    .tp_basicsize = sizeof(FloatArray),
    .tp_new = (newfunc) FloatArray_new,
    .tp_dealloc = (destructor) FloatArray_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_as_sequence = &FloatArrayTypeAsSeq
};

static FloatArray * create_float_array(size_t count) {
    FloatArray * array = (FloatArray *) PyObject_CallObject((PyObject *) &FloatArrayType, NULL);
    if (array != NULL) {
        array->count = count;
        array->elements = malloc(sizeof(double) * count);
    }
    return array;
}

// Mandelbrot =================================================================
static PyObject * mandelbrot_native(PyObject * self, PyObject * args) {
    size_t w = 0;
    size_t h = 0;
    size_t row_start = 0;
    size_t slice_height = 0;
    double re_start;
    double re_end;
    double im_start;
    double im_end;
    double escape_radius;
    size_t max_iter;
    double p;
    int get_int_data;
    int get_float_data;
    int get_edges;
    if (!PyArg_ParseTuple(args, "nnnndddddndiii",
            &w, &h, &row_start, &slice_height,
            &re_start, &re_end, &im_start, &im_end, &escape_radius,
            &max_iter, &p, &get_int_data, &get_float_data, &get_edges)) {
        return NULL;
    }

    const size_t data_count = w * slice_height;
    IntArray * int_data = NULL;
    if (get_int_data) {
        int_data = create_int_array(data_count);
        if (int_data == NULL) {
            return NULL;
        }
    }
    FloatArray * float_data = NULL;
    if (get_float_data) {
        float_data = create_float_array(data_count);
        if (float_data == NULL) {
            return NULL;
        }
    }

    PyObject * edges = NULL;
    if (get_edges) {
        edges = PyList_New(0);
        if (edges == NULL) {
            return NULL;
        }
    }

    const size_t row_end = row_start + slice_height;
    for (size_t col = 0; col < w; col++) {
        for (size_t row = row_start; row < row_end; row++) {
            const double x = re_start + ((double)col) / ((double)w) * (re_end - re_start);
            const double y = im_start + ((double)row) / ((double)h) * (im_end - im_start);
            const double complex c = x + y * I;
            double complex z = 0;
            size_t n = 0;
            while (cabs(z) <= escape_radius && n < max_iter) {
                z = cpow(z, p) + c;
                n++;
            }

            const size_t index = (row - row_start) * w + col;
            if (get_int_data) {
                int_data->elements[index] = n;
            }

            if (get_float_data) {
                double n_float = n;
                if (n < max_iter) {
                    n_float =- log(log(cabs(z)) / log(escape_radius)) / log(p);
                }
                float_data->elements[index] = n_float;
            }

            if (get_edges && n == (max_iter - 1)) {
                PyObject * edge = Py_BuildValue("(dd)", x, y);
                if (edge == NULL) {
                    return NULL;
                }
                if (PyList_Append(edges, edge) < 0) {
                    return NULL;
                }
            }
        }
    }

    if (int_data == NULL) {
        int_data = get_none_value();
    }
    if (float_data == NULL) {
        float_data = get_none_value();
    }
    if (edges == NULL) {
        edges = get_none_value();
    }
    return Py_BuildValue("(OOO)", int_data, float_data, edges);
}

static PyMethodDef mandelbrot_native_methods[] = {
    {"mandelbrot_native", mandelbrot_native, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef mandelbrot_native_module = {
    PyModuleDef_HEAD_INIT, "mandelbrot_native", NULL, -1, mandelbrot_native_methods
};


static bool add_array_type(PyObject * module, const char * type_name,
        PyTypeObject * type) {
    if (PyType_Ready(type) < 0) {
        Py_DECREF(module);
        return true;
    }
    if (PySequence_Check((PyObject *) type) < 0) {
        Py_DECREF(module);
        return true;
    }

    if (PyModule_AddObjectRef(module, type_name, (PyObject *) type) < 0) {
        Py_DECREF(module);
        return true;
    }

    return false;
}

PyMODINIT_FUNC PyInit_mandelbrot_native() {
    PyObject * module = PyModule_Create(&mandelbrot_native_module);
    if (module == NULL) {
        return NULL;
    }

    if (add_array_type(module, "IntArray", &IntArrayType)) {
        return NULL;
    }

    if (add_array_type(module, "FloatArray", &FloatArrayType)) {
        return NULL;
    }

    return module;
}
