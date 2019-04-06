import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output, Image, display, HTML


def remove_3d_panes(ax):
    ax.w_xaxis.set_pane_color((1., 1., 1., 0.))
    ax.w_yaxis.set_pane_color((1., 1., 1., 0.))
    ax.w_zaxis.set_pane_color((1., 1., 1., 0.))


def remove_3d_lines(ax):
    ax.w_xaxis.line.set_color((1., 1., 1., 0.))
    ax.w_yaxis.line.set_color((1., 1., 1., 0.))
    ax.w_zaxis.line.set_color((1., 1., 1., 0.))


def remove_3d_accessoires(ax):
    remove_3d_panes(ax)
    remove_3d_lines(ax)
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.zaxis.set_ticks([])


def plot_3d_axes(ax, xmin=0, xmax=1, ymin=0, ymax=1, zmin=0, zmax=1):
    ax.plot([xmin, xmax], [ymax, ymax], [zmin, zmin], color='black', zorder=0, lw=1)
    ax.plot([xmin, xmin], [ymin, ymax], [zmin, zmin], color='black', zorder=0, lw=1)
    ax.plot([xmin, xmin], [ymax, ymax], [zmin, zmax], color='black', zorder=0, lw=1)


def label_3d_axes(ax, xmin=0, xmax=1, ymin=0, ymax=1, zmin=0, zmax=1):
    ax.text(xmax, ymax, zmin, '$x_1$', None, fontsize=20)
    ax.text(xmin, ymax, zmin, '$x_2$', None, fontsize=20)
    ax.text(xmin, ymax, zmax, '$\\mathcal{C}(x_1, x_2)$', None, fontsize=20)


def set_tick_size(ax, size=20):
    ax.xaxis.set_tick_params(labelsize=size)
    ax.yaxis.set_tick_params(labelsize=size)


def pcm(_ax, _m, centering=False):
    shp = _m.shape
    if len(shp) > 2:
        _indices = np.zeros((len(shp) - 2,), np.int)
        _m = _m[_indices].reshape(shp[-2:])
    if centering:
        v = np.max(np.abs(_m))
        vmin, vmax = -v, v
    else:
        vmin, vmax = None, None
    p = _ax.pcolormesh(_m.T, cmap='magma', vmin=vmin, vmax=vmax)
    plt.colorbar(p, ax=_ax)


def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add() 
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = "<stripped %d bytes>"%size
    return strip_def


def show_graph(graph_def, max_const_size=32):
    """Visualize TensorFlow graph."""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph'+str(np.random.rand()))

    iframe = """
        <iframe seamless style="width:1200px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    display(HTML(iframe))

