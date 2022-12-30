import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from patbert.common import vis
plt.style.use('dark_background')

class EmbeddingFigure():
    def __init__(self, num_vectors, num_sub_vectors, num_sub_sub_vectors,  len, sub_len, sub_sub_len, 
            num_sss_vecs=0, sss_len=0, s_vec=20, s_sub_vec=4, s_sub_sub_vec=.1, s_sss_vec=.01,
            modalities = ['Disease', 'Medicine', 'Lab Test', 'Procedure', 'SEP', 'CLS', 'MASK'], seed=0,
            figsize=(10,10), dpi=100, fontsize=12, save_path=None, marker_multiplier=8, axins_kwargs=None,
            zoom_sub_vec_num = 9, text_x_shift=0, text_y_shift=0, add_inset=True):
        self.rng = np.random.default_rng(seed)
        self.num_vectors = num_vectors 
        self.num_sub_vectors = num_sub_vectors
        self.num_sub_sub_vectors = num_sub_sub_vectors
        self.num_sss_vecs = num_sss_vecs
        self.len = len
        self.sub_len = sub_len
        self.sub_sub_len = sub_sub_len
        self.sss_len = sss_len
        self.modalities = modalities
        self.text_x_shift = text_x_shift
        self.text_y_shift = text_y_shift
        self.figsize = figsize
        self.dpi = dpi
        self.fontsize = fontsize
        self.save_path = save_path
        self.s_vec = s_vec
        self.s_sub_vec = s_sub_vec
        self.s_sub_sub_vec = s_sub_sub_vec
        self.s_sss_vec = s_sss_vec
        self.vecs = None
        self.sub_vecs = None
        self.sub_sub_vecs = None
        self.sss_vecs = None
        self.axins_kwargs = axins_kwargs
        self.default_colors = self.get_nice_colors_dark_background()
        self.sub_sub_vector_colors = None 
        self.sub_sub_vector_colors =  None 
        self.sss_vector_colors = None
        self.get_vector_colors()
        self.marker_multiplier = marker_multiplier
        self.axins = None
        self.zoom_sub_vec_num = zoom_sub_vec_num
        self.add_inset = add_inset

    def __call__(self):
        assert len(self.modalities) == self.num_vectors, 'Number of modalities must match number of vectors'   
        self.default_colors = self.default_colors*5
        if self.num_sss_vecs > 0:
            self.vecs, self.sub_vecs, self.sub_sub_vecs, self.sss_vecs = self.get_vector_hierarchy()
        else:
            self.vecs, self.sub_vecs, self.sub_sub_vecs = self.get_vector_hierarchy()
        
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        self.scatter_vecs(self.ax)
        for i in range(self.num_vectors):
            self.ax.text(self.vecs[i,0]+self.text_x_shift, self.vecs[i,1]+self.text_y_shift, 
                self.modalities[i], fontsize=self.fontsize, color='k',
                bbox=dict(boxstyle="round",
                   
                   fc=(.8, 0.8, 0.8),
                   alpha=.8,
                   ))
        
        if self.add_inset:
            if self.zoom_sub_vec_num:
                self.reset_zoom_location()
            self.axins = vis.add_zoom_inset(self.ax,bbox_transform=self.ax.transAxes, **self.axins_kwargs)
            self.scatter_vecs(self.axins, marker_multiplier=self.marker_multiplier)
        # save figure
        return self.fig, self.ax, self.axins

    def save_fig(self):
        self.fig.savefig(self.save_path, dpi=self.dpi,  bbox_inches='tight')

    def reset_zoom_location(self):
        sub_vec_x = self.sub_vecs[self.zoom_sub_vec_num,0]
        sub_vec_y = self.sub_vecs[self.zoom_sub_vec_num,1]
        x_width = abs(self.axins_kwargs['xlim'][1] - self.axins_kwargs['xlim'][0])
        self.axins_kwargs['xlim'] = (sub_vec_x-x_width/2, sub_vec_x+x_width/2)
        y_width = abs(self.axins_kwargs['ylim'][1] - self.axins_kwargs['ylim'][0])
        self.axins_kwargs['ylim'] = (sub_vec_y-y_width/2, sub_vec_y+y_width/2)

    def get_vector_colors(self):
        self.sub_vector_color = np.repeat(self.default_colors, self.num_sub_vectors)
        self.sub_sub_vector_colors = np.repeat(self.default_colors, self.num_sub_vectors*self.num_sub_sub_vectors)
        if self.num_sss_vecs > 0:
            self.sss_vector_colors = np.repeat(self.default_colors, self.num_sub_vectors*self.num_sub_sub_vectors*self.num_sss_vecs)
        
    def scatter_vecs(self, ax, marker_multiplier=1):
        ax.scatter(self.vecs[:,0], self.vecs[:,1], s=self.s_vec*marker_multiplier, color=self.default_colors[:self.num_vectors])
        ax.scatter(self.sub_vecs[:,0], self.sub_vecs[:,1], s=self.s_sub_vec*marker_multiplier, color=self.sub_vector_color[:self.num_sub_vectors*self.num_vectors])
        ax.scatter(self.sub_sub_vecs[:,0], self.sub_sub_vecs[:,1], s=self.s_sub_sub_vec*marker_multiplier, 
            color=self.sub_sub_vector_colors[:self.num_sub_sub_vectors*self.num_sub_vectors*self.num_vectors])
        if self.num_sss_vecs > 0:
            ax.scatter(self.sss_vecs[:,0], self.sss_vecs[:,1], s=self.s_sss_vec*marker_multiplier, 
                color=self.sss_vector_colors[:self.num_sss_vecs*self.num_sub_sub_vectors*self.num_sub_vectors*self.num_vectors])

    def get_vector_hierarchy(self):
        all_vecs = []
        vectors = self.get_vec(self.num_vectors,self.len)
        sub_vec = []
        sub_sub_vec = []
        sss_vec = []
        for i in range(self.num_vectors):
            sub_vec.append(vectors[i] + self.get_vec(self.num_sub_vectors, self.sub_len))
            for j in range(self.num_sub_vectors):
                sub_sub_vec.append(sub_vec[i][j] + self.get_vec(self.num_sub_sub_vectors,self.sub_sub_len))
       
        sub_vec = np.concatenate(sub_vec, axis=0)
        sub_sub_vec = np.concatenate(sub_sub_vec, axis=0)
        if self.num_sss_vecs > 0:
            for k in range(len(sub_sub_vec)):
                sss_vec.append(sub_sub_vec[k] + self.get_vec(self.num_sss_vecs, self.sss_len))
        if self.num_sss_vecs > 0:
            sss_vec = np.concatenate(sss_vec, axis=0)
            return vectors, sub_vec, sub_sub_vec, sss_vec
        else:
            return vectors, sub_vec, sub_sub_vec
            
    def get_vec(self, n, len):
        vec = self.rng.uniform(-1,1,(n,2))
        vec = len*vec/la.norm(vec, axis=1).reshape(-1,1)
        return vec
    @staticmethod
    def get_nice_colors_dark_background():
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        add_colors = ['#45bcde', '#52d273', '#e94f64', '#e57254']
        for color in add_colors:
            colors.insert(0, color)
        return  colors
    def turn_off_frame(self):
        self.ax.axes.get_xaxis().set_visible(False)
        self.ax.axes.get_yaxis().set_visible(False)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['bottom'].set_visible(False)
        self.ax.spines['left'].set_visible(False)
        return self.ax