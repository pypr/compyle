import pytest
import numpy as np

from ..array import Array
from ..config import get_config
import compyle.array as array


check_all_backends = pytest.mark.parametrize('backend',
                                             ['cython', 'opencl', 'cuda'])


def make_dev_array(backend, n=16):
    dev_array = Array(np.int32, n=n, backend=backend)
    dev_array.fill(0)
    dev_array[0] = 1
    return dev_array


def check_import(backend):
    if backend == 'opencl':
        pytest.importorskip('pyopencl')
    if backend == 'cuda':
        pytest.importorskip('pycuda')


@check_all_backends
def test_reserve(backend):
    check_import(backend)

    # Given
    dev_array = make_dev_array(backend)

    # When
    dev_array.reserve(64)

    # Then
    assert len(dev_array.get_data()) == 64
    assert dev_array.length == 16
    assert dev_array[0] == 1


@check_all_backends
def test_resize_with_reallocation(backend):
    check_import(backend)

    # Given
    dev_array = make_dev_array(backend)

    # When
    dev_array.resize(64)

    # Then
    assert len(dev_array.get_data()) == 64
    assert dev_array.length == 64
    assert dev_array[0] == 1


@check_all_backends
def test_resize_without_reallocation(backend):
    check_import(backend)

    # Given
    dev_array = make_dev_array(backend, n=128)

    # When
    dev_array.resize(64)

    # Then
    assert len(dev_array.get_data()) == 128
    assert dev_array.length == 64
    assert dev_array[0] == 1


@check_all_backends
def test_copy(backend):
    check_import(backend)

    # Given
    dev_array = make_dev_array(backend)

    # When
    dev_array_copy = dev_array.copy()

    # Then
    print(dev_array.dev, dev_array_copy.dev)
    assert np.all(dev_array.get() == dev_array_copy.get())

    dev_array_copy[0] = 2
    assert dev_array[0] != dev_array_copy[0]


@check_all_backends
def test_append_with_reallocation(backend):
    check_import(backend)

    # Given
    dev_array = make_dev_array(backend)

    # When
    dev_array.append(2)

    # Then
    assert dev_array[-1] == 2
    assert len(dev_array.get_data()) == 32


@check_all_backends
def test_append_without_reallocation(backend):
    check_import(backend)

    # Given
    dev_array = make_dev_array(backend)
    dev_array.reserve(20)

    # When
    dev_array.append(2)

    # Then
    assert dev_array[-1] == 2
    assert len(dev_array.get_data()) == 20


@check_all_backends
def test_extend(backend):
    check_import(backend)

    # Given
    dev_array = make_dev_array(backend)
    new_array = 2 + array.zeros(64, dtype=np.int32, backend=backend)

    # When
    dev_array.extend(new_array)

    # Then
    old_nparr = dev_array.get()
    new_nparr = new_array.get()
    assert np.all(old_nparr[-len(new_array)] == new_nparr)


@check_all_backends
def test_remove(backend):
    check_import(backend)

    # Given
    dev_array = Array(np.int32, backend=backend)
    orig_array = array.arange(0, 16, 1, dtype=np.int32,
                              backend=backend)
    dev_array.set_data(orig_array)
    indices = array.arange(0, 8, 1, dtype=np.int32, backend=backend)

    # When
    dev_array.remove(indices)

    # Then
    assert np.all(dev_array.get() == (8 + indices).get())


@check_all_backends
def test_align(backend):
    check_import(backend)

    # Given
    dev_array = Array(np.int32, backend=backend)
    orig_array = array.arange(0, 16, 1, dtype=np.int32, backend=backend)
    dev_array.set_data(orig_array)
    indices = array.arange(15, -1, -1, dtype=np.int32, backend=backend)

    # When
    dev_array = dev_array.align(indices)

    # Then
    assert np.all(dev_array.get() == indices.get())


@check_all_backends
def test_align_multiple(backend):
    check_import(backend)

    # Given
    dev_array_a = Array(np.uint32, backend=backend)
    dev_array_b = Array(np.float32, backend=backend)
    orig_array_a = array.arange(0, 1024, 1, dtype=np.uint32, backend=backend)
    orig_array_b = array.arange(
        1024, 2048, 1, dtype=np.float32, backend=backend)
    dev_array_a.set_data(orig_array_a)
    dev_array_b.set_data(orig_array_b)

    indices = array.arange(1023, -1, -1, dtype=np.int64, backend=backend)

    # When
    dev_array_a, dev_array_b = array.align([dev_array_a, dev_array_b],
                                           indices)

    # Then
    assert np.all(dev_array_a.get() == indices.get())
    assert np.all(dev_array_b.get() - 1024 == indices.get())


@check_all_backends
def test_squeeze(backend):
    check_import(backend)

    # Given
    dev_array = make_dev_array(backend)
    dev_array.fill(2)
    dev_array.reserve(32)
    assert dev_array.alloc == 32

    # When
    dev_array.squeeze()

    # Then
    assert dev_array.alloc == 16


@check_all_backends
def test_copy_values(backend):
    check_import(backend)

    # Given
    dev_array = make_dev_array(backend)
    dev_array.fill(2)

    dest = array.empty(8, dtype=np.int32, backend=backend)
    indices = array.arange(0, 8, 1, dtype=np.int32, backend=backend)

    # When
    dev_array.copy_values(indices, dest)

    # Then
    assert np.all(dev_array[:len(indices)].get() == dest.get())


@check_all_backends
def test_min_max(backend):
    check_import(backend)

    # Given
    dev_array = make_dev_array(backend)
    dev_array.fill(2)
    dev_array[0], dev_array[1] = 1, 10

    # When
    dev_array.update_min_max()

    # Then
    assert dev_array.minimum == 1
    assert dev_array.maximum == 10


@check_all_backends
def test_sort_by_keys(backend):
    check_import(backend)

    # Given
    nparr1 = np.random.randint(0, 100, 16, dtype=np.int32)
    nparr2 = np.random.randint(0, 100, 16, dtype=np.int32)
    dev_array1, dev_array2 = array.wrap(nparr1, nparr2, backend=backend)

    # When
    out_array1, out_array2 = array.sort_by_keys([dev_array1, dev_array2])

    # Then
    order = np.argsort(nparr1)
    act_result1 = np.take(nparr1, order)
    act_result2 = np.take(nparr2, order)
    assert np.all(out_array1.get() == act_result1)
    assert np.all(out_array2.get() == act_result2)


def test_radix_sort_by_keys():
    backend = 'cython'
    for use_openmp in [True, False]:
        get_config().use_openmp = use_openmp
        # Given
        nparr1 = np.random.randint(0, 100, 16, dtype=np.int32)
        nparr2 = np.random.randint(0, 100, 16, dtype=np.int32)
        dev_array1, dev_array2 = array.wrap(nparr1, nparr2, backend=backend)

        # When
        out_array1, out_array2 = array.sort_by_keys([dev_array1, dev_array2],
                                                    use_radix_sort=True)

        # Then
        order = np.argsort(nparr1)
        act_result1 = np.take(nparr1, order)
        act_result2 = np.take(nparr2, order)
        assert np.all(out_array1.get() == act_result1)
        assert np.all(out_array2.get() == act_result2)
    get_config().use_openmp = False


@pytest.mark.parametrize(
    'backend', ['cython', 'opencl',
                pytest.param('cuda', marks=pytest.mark.xfail)])
def test_sort_by_keys_with_output(backend):
    check_import(backend)

    # Given
    nparr1 = np.random.randint(0, 100, 16, dtype=np.int32)
    nparr2 = np.random.randint(0, 100, 16, dtype=np.int32)
    dev_array1, dev_array2 = array.wrap(nparr1, nparr2, backend=backend)
    out_arrays = [
        array.zeros_like(dev_array1),
        array.zeros_like(dev_array2)]

    # When
    array.sort_by_keys([dev_array1, dev_array2],
                       out_list=out_arrays, use_radix_sort=False)

    # Then
    order = np.argsort(nparr1)
    act_result1 = np.take(nparr1, order)
    act_result2 = np.take(nparr2, order)
    assert np.all(out_arrays[0].get() == act_result1)
    assert np.all(out_arrays[1].get() == act_result2)


@check_all_backends
def test_dot(backend):
    check_import(backend)

    # Given
    a = make_dev_array(backend)
    a.fill(1)

    b = make_dev_array(backend)
    b.fill(2)

    # When
    out_array = array.dot(a, b)

    # Then
    assert np.all(out_array == 32)


@check_all_backends
def test_cumsum(backend):
    check_import(backend)

    # Given
    a = array.ones(100, dtype=int, backend=backend)

    # When
    b = array.cumsum(a)

    # Then
    a.pull()
    b.pull()
    assert np.all(b.data == np.cumsum(a.data))
