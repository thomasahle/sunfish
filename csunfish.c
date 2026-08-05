#define PY_SSIZE_T_CLEAN
#include "csunfish.h"
#include <stdbool.h>
#include <string.h>

// Forward declaration of custom types
typedef struct {
    PyObject_HEAD
    Position pos;
} PyPositionObject;

typedef struct {
    PyObject_HEAD
    Searcher *searcher;
} PySearcherObject;

typedef struct {
    int i;
    int j;
    char prom;
} PyMoveData;

// Constants and utility

static PyObject *csunfish_version(PyObject *self, PyObject *args) {
    return PyUnicode_FromString(sf_version);
}

static PyObject *csunfish_MATE_LOWER(PyObject *self, PyObject *args) {
    return PyLong_FromLong(MATE_LOWER);
}

static PyObject *csunfish_MATE_UPPER(PyObject *self, PyObject *args) {
    return PyLong_FromLong(MATE_UPPER);
}

static PyObject *csunfish_opt_ranges(PyObject *self, PyObject *args) {
    // Hardcode based on python code
    // opt_ranges = dict(QS=(0,300), QS_A=(0,300), EVAL_ROUGHNESS=(0,50))
    PyObject *d = PyDict_New();
    PyObject *qs_range = Py_BuildValue("(ii)", 0, 300);
    PyObject *qs_a_range = Py_BuildValue("(ii)", 0, 300);
    PyObject *er_range = Py_BuildValue("(ii)", 0, 50);
    PyDict_SetItemString(d, "QS", qs_range);
    PyDict_SetItemString(d, "QS_A", qs_a_range);
    PyDict_SetItemString(d, "EVAL_ROUGHNESS", er_range);
    Py_DECREF(qs_range);
    Py_DECREF(qs_a_range);
    Py_DECREF(er_range);
    return d;
}

static PyObject *csunfish_parse(PyObject *self, PyObject *args) {
    const char *sq;
    if (!PyArg_ParseTuple(args, "s", &sq))
        return NULL;
    int idx = sf_parse(sq);
    return PyLong_FromLong(idx);
}

static PyObject *csunfish_render(PyObject *self, PyObject *args) {
    int i;
    if (!PyArg_ParseTuple(args, "i", &i))
        return NULL;
    char out[3];
    sf_render(i, out);
    return PyUnicode_FromString(out);
}

// Move object as a factory function
// We'll return a Python object with attributes i,j,prom
static PyObject *csunfish_Move(PyObject *self, PyObject *args) {
    int i, j;
    const char *prom = "";
    if (!PyArg_ParseTuple(args, "ii|s", &i, &j, &prom))
        return NULL;

    PyObject *dict = PyDict_New();
    PyDict_SetItemString(dict, "i", PyLong_FromLong(i));
    PyDict_SetItemString(dict, "j", PyLong_FromLong(j));
    PyDict_SetItemString(dict, "prom", PyUnicode_FromString(prom));
    return dict;
}

// Position type

static void PyPosition_dealloc(PyPositionObject *self) {
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* PyPosition_move(PyPositionObject *self, PyObject *args) {
    PyObject *move_dict;
    if (!PyArg_ParseTuple(args, "O", &move_dict))
        return NULL;
    PyObject *i_obj = PyDict_GetItemString(move_dict, "i");
    PyObject *j_obj = PyDict_GetItemString(move_dict, "j");
    PyObject *prom_obj = PyDict_GetItemString(move_dict, "prom");
    if (!i_obj || !j_obj || !prom_obj) {
        PyErr_SetString(PyExc_ValueError, "Invalid move dict");
        return NULL;
    }
    int i = (int)PyLong_AsLong(i_obj);
    int j = (int)PyLong_AsLong(j_obj);
    const char *prom = PyUnicode_AsUTF8(prom_obj);

    Move m;
    sf_move_make(i, j, prom, &m);
    Position newpos = sf_position_move(&self->pos, m);

    PyPositionObject *res = PyObject_New(PyPositionObject, (PyTypeObject*)Py_TYPE(self));
    res->pos = newpos;
    return (PyObject*)res;
}

static PyObject* PyPosition_rotate(PyPositionObject *self, PyObject *args) {
    Position newpos = sf_position_rotate(&self->pos, false);
    PyPositionObject *res = PyObject_New(PyPositionObject, Py_TYPE(self));
    res->pos = newpos;
    return (PyObject*)res;
}

static PyObject* PyPosition_gen_moves(PyPositionObject *self, PyObject *args) {
    Move moves[256];
    int count=0;
    sf_position_gen_moves(&self->pos, moves, &count);
    PyObject *list = PyList_New(count);
    for (int k=0; k<count; k++) {
        PyObject *dict = PyDict_New();
        PyDict_SetItemString(dict, "i", PyLong_FromLong(moves[k].i));
        PyDict_SetItemString(dict, "j", PyLong_FromLong(moves[k].j));
        char prom_str[2] = {0,0};
        if (moves[k].prom) prom_str[0] = moves[k].prom;
        PyDict_SetItemString(dict, "prom", PyUnicode_FromString(prom_str));
        PyList_SetItem(list, k, dict);
    }
    return list;
}

static PyObject* PyPosition_score(PyPositionObject *self, void *closure) {
    int sc = sf_position_score(&self->pos);
    return PyLong_FromLong(sc);
}

static PyObject* PyPosition_board(PyPositionObject *self, void *closure) {
    const char* b = sf_position_board(&self->pos);
    return PyUnicode_FromString(b);
}

static PyObject* PyPosition_kp(PyPositionObject *self, void *closure) {
    int k = sf_position_kp(&self->pos);
    return PyLong_FromLong(k);
}

static PyGetSetDef PyPosition_getset[] = {
    {"score", (getter)PyPosition_score, NULL, "score of position", NULL},
    {"board", (getter)PyPosition_board, NULL, "board string", NULL},
    {"kp",    (getter)PyPosition_kp,    NULL, "king passant square", NULL},
    {NULL}
};

static PyMethodDef PyPosition_methods[] = {
    {"move", (PyCFunction)PyPosition_move, METH_VARARGS, "Make a move"},
    {"rotate", (PyCFunction)PyPosition_rotate, METH_NOARGS, "Rotate the position"},
    {"gen_moves", (PyCFunction)PyPosition_gen_moves, METH_NOARGS, "Generate moves"},
    {NULL,NULL,0,NULL}
};

static PyTypeObject PyPositionType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "csunfish.Position",
    .tp_basicsize = sizeof(PyPositionObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_dealloc = (destructor)PyPosition_dealloc,
    .tp_methods = PyPosition_methods,
    .tp_getset = PyPosition_getset,
};

// Position constructor: Position(board, score, wc, bc, ep, kp)
// wc and bc are tuples of bool
static PyObject *csunfish_Position(PyObject *self, PyObject *args) {
    const char* board;
    int score, ep, kp;
    PyObject *wc_obj, *bc_obj;
    if (!PyArg_ParseTuple(args, "siOOii", &board, &score, &wc_obj, &bc_obj, &ep, &kp))
        return NULL;

    bool wc0 = PyObject_IsTrue(PyTuple_GetItem(wc_obj,0));
    bool wc1 = PyObject_IsTrue(PyTuple_GetItem(wc_obj,1));
    bool bc0 = PyObject_IsTrue(PyTuple_GetItem(bc_obj,0));
    bool bc1 = PyObject_IsTrue(PyTuple_GetItem(bc_obj,1));

    Position pos = sf_position_create(board, score, wc0, wc1, bc0, bc1, ep, kp);

    PyPositionObject *pobj = PyObject_New(PyPositionObject, &PyPositionType);
    pobj->pos = pos;
    return (PyObject*)pobj;
}

// Searcher type

static void PySearcher_dealloc(PySearcherObject *self) {
    sf_searcher_free(self->searcher);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* PySearcher_bound(PySearcherObject *self, PyObject *args) {
    PyObject *pos_obj;
    int gamma, depth;
    if (!PyArg_ParseTuple(args, "Oii", &pos_obj, &gamma, &depth))
        return NULL;
    if (!PyObject_TypeCheck(pos_obj, &PyPositionType)) {
        PyErr_SetString(PyExc_TypeError, "pos must be a Position");
        return NULL;
    }
    PyPositionObject *p = (PyPositionObject*)pos_obj;
    int score = sf_searcher_bound(self->searcher, p->pos, gamma, depth, true);
    return PyLong_FromLong(score);
}

static PyObject* PySearcher_nodes(PySearcherObject *self, PyObject *args) {
    return PyLong_FromLong(sf_searcher_nodes(self->searcher));
}

static PyObject* PySearcher_tp_move_get(PySearcherObject *self, PyObject *args) {
    PyObject *pos_obj;
    if (!PyArg_ParseTuple(args, "O", &pos_obj))
        return NULL;
    if (!PyObject_TypeCheck(pos_obj, &PyPositionType)) {
        PyErr_SetString(PyExc_TypeError, "pos must be a Position");
        return NULL;
    }
    PyPositionObject *p = (PyPositionObject*)pos_obj;
    Move m;
    bool found = sf_searcher_tp_move_get(self->searcher, &p->pos, &m);
    if (!found) {
        Py_RETURN_NONE;
    }
    PyObject *dict = PyDict_New();
    PyDict_SetItemString(dict, "i", PyLong_FromLong(m.i));
    PyDict_SetItemString(dict, "j", PyLong_FromLong(m.j));
    char prom_str[2] = {0,0};
    if (m.prom) prom_str[0] = m.prom;
    PyDict_SetItemString(dict, "prom", PyUnicode_FromString(prom_str));
    return dict;
}

static PyObject* PySearcher_search(PySearcherObject *self, PyObject *args) {
    // args: hist is a Python list of Positions
    // We'll convert hist to a C array of Positions
    PyObject *hist_list;
    if (!PyArg_ParseTuple(args, "O", &hist_list))
        return NULL;
    if (!PyList_Check(hist_list)) {
        PyErr_SetString(PyExc_TypeError, "hist must be a list of Positions");
        return NULL;
    }
    int length = (int)PyList_Size(hist_list);
    Position *positions = (Position*)malloc(sizeof(Position)*length);
    for (int i=0; i<length; i++) {
        PyObject *item = PyList_GetItem(hist_list, i);
        if (!PyObject_TypeCheck(item, &PyPositionType)) {
            free(positions);
            PyErr_SetString(PyExc_TypeError, "hist elements must be Positions");
            return NULL;
        }
        PyPositionObject *p = (PyPositionObject*)item;
        positions[i] = p->pos;
    }

    // Set history in searcher
    sf_searcher_set_history(self->searcher, positions, length);

    int count = 0;
    int *depths=NULL,*gammas=NULL,*scores=NULL;
    Move *moves=NULL;
    // Perform search
    // This is a custom function you must implement in sunfish.c:
    // int sf_searcher_search(Searcher *s, const Position *history, int hist_len,
    //                        int *count, int **depths, int **gammas, int **scores, Move **moves);
    if (sf_searcher_search(self->searcher, positions, length, &count, &depths, &gammas, &scores, &moves) != 0) {
        free(positions);
        PyErr_SetString(PyExc_RuntimeError, "search failed");
        return NULL;
    }
    free(positions);

    PyObject *res_list = PyList_New(count);
    for (int i=0; i<count; i++) {
        PyObject *mv_dict = PyDict_New();
        PyDict_SetItemString(mv_dict, "i", PyLong_FromLong(moves[i].i));
        PyDict_SetItemString(mv_dict, "j", PyLong_FromLong(moves[i].j));
        char prom_str[2] = {0};
        if (moves[i].prom) prom_str[0]=moves[i].prom;
        PyDict_SetItemString(mv_dict, "prom", PyUnicode_FromString(prom_str));
        PyObject *tpl = Py_BuildValue("(iiiO)", depths[i], gammas[i], scores[i], mv_dict);
        Py_DECREF(mv_dict);
        PyList_SetItem(res_list, i, tpl);
    }
    free(depths);free(gammas);free(scores);free(moves);

    return res_list;
}

static PyMethodDef PySearcher_methods[] = {
    {"bound", (PyCFunction)PySearcher_bound, METH_VARARGS, "bound(pos,gamma,depth)"},
    {"nodes", (PyCFunction)PySearcher_nodes, METH_NOARGS, "nodes()"},
    {"tp_move_get", (PyCFunction)PySearcher_tp_move_get, METH_VARARGS, "tp_move_get(pos)"},
    {"search", (PyCFunction)PySearcher_search, METH_VARARGS, "search(hist)"},
    {NULL,NULL,0,NULL}
};

static PyTypeObject PySearcherType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "csunfish.Searcher",
    .tp_basicsize = sizeof(PySearcherObject),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_dealloc = (destructor)PySearcher_dealloc,
    .tp_methods = PySearcher_methods,
};

// Searcher constructor
static PyObject *csunfish_Searcher(PyObject *self, PyObject *args) {
    PySearcherObject *obj = PyObject_New(PySearcherObject, &PySearcherType);
    obj->searcher = sf_searcher_create();
    return (PyObject*)obj;
}

// Module methods
static PyMethodDef csunfish_methods[] = {
    {"version", (PyCFunction)csunfish_version, METH_NOARGS, "Get version"},
    {"MATE_LOWER", (PyCFunction)csunfish_MATE_LOWER, METH_NOARGS, "MATE_LOWER constant"},
    {"MATE_UPPER", (PyCFunction)csunfish_MATE_UPPER, METH_NOARGS, "MATE_UPPER constant"},
    {"opt_ranges", (PyCFunction)csunfish_opt_ranges, METH_NOARGS, "Get option ranges"},
    {"parse", (PyCFunction)csunfish_parse, METH_VARARGS, "parse square string"},
    {"render", (PyCFunction)csunfish_render, METH_VARARGS, "render square index"},
    {"Move", (PyCFunction)csunfish_Move, METH_VARARGS, "Create a move dict(i,j,prom)"},
    {"Position", (PyCFunction)csunfish_Position, METH_VARARGS, "Create a position"},
    {"Searcher", (PyCFunction)csunfish_Searcher, METH_NOARGS, "Create a searcher"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef csunfish_module = {
    PyModuleDef_HEAD_INIT,
    "csunfish",
    "C interface to sunfish engine",
    -1,
    csunfish_methods
};

PyMODINIT_FUNC PyInit_csunfish(void) {
    if (PyType_Ready(&PyPositionType) < 0)
        return NULL;
    if (PyType_Ready(&PySearcherType) < 0)
        return NULL;

    PyObject *m = PyModule_Create(&csunfish_module);
    if (!m) return NULL;

    Py_INCREF(&PyPositionType);
    if (PyModule_AddObject(m, "Position", (PyObject*)&PyPositionType) < 0) {
        Py_DECREF(&PyPositionType);
        Py_DECREF(m);
        return NULL;
    }

    Py_INCREF(&PySearcherType);
    if (PyModule_AddObject(m, "Searcher", (PyObject*)&PySearcherType) < 0) {
        Py_DECREF(&PySearcherType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
