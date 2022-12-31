import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from patbert.common import vis


class EmbeddingFigure():
    def __init__(self,vector_nums, vector_lens, marker_sizes,
            modalities = ['Disease', 'Medicine',  'Procedure',  'SEP', 'Lab Test', 'CLS', 'MASK'], seed=0,
            figsize=(10,10), dpi=100, fontsize=12, save_path=None, marker_multiplier=8, axins_kwargs=None,
            zoom_sub_vec_num = 9, text_x_shift=0, text_y_shift=0, add_inset=True, style='dark_background'):
        self.style = style
        plt.rcParams.update(plt.rcParamsDefault)
        plt.style.use(style)
        self.rng = np.random.default_rng(seed)
        self.vec_nums = vector_nums
        self.vector_lens = vector_lens
        self.modalities = modalities
        self.text_x_shift = text_x_shift
        self.text_y_shift = text_y_shift
        self.figsize = figsize
        self.dpi = dpi
        self.fontsize = fontsize
        self.save_path = save_path
        self.marker_sizes = marker_sizes
   
        self.vecs = [] # list of arrays of vectors
        self.axins_kwargs = axins_kwargs
        self.def_colors = self.get_nice_colors_dark_background()
        self.get_vec_colors()
        self.marker_multiplier = marker_multiplier
        self.axins = None
        self.zoom_sub_vec_num = zoom_sub_vec_num
        self.add_inset = add_inset

    def __call__(self):
        assert len(self.modalities) == self.vec_nums[0], 'Number of modalities must match number of vectors'   
        self.def_colors = self.def_colors*5
        self.vecs = self.get_vector_hierarchy()
            
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        self.scatter_vecs_for(self.ax)
        for i in range(self.vec_nums[0]):
            self.ax.text(self.vecs[0][i,0]+self.text_x_shift, self.vecs[0][i,1]+self.text_y_shift, 
                self.modalities[i], fontsize=self.fontsize, color='k',
                bbox=dict(boxstyle="round",
                   fc=(.8, 0.8, 0.8),
                   alpha=.8,
                   ))
        
        if self.add_inset:
            if self.zoom_sub_vec_num:
                self.reset_zoom_location()
            self.axins = vis.add_zoom_inset(self.ax, bbox_transform=self.ax.transAxes, **self.axins_kwargs)
            self.scatter_vecs_for(self.axins, marker_multiplier=self.marker_multiplier)
        # save figure
        return self.fig, self.ax, self.axins

    def save_fig(self):
        self.fig.savefig(self.save_path, dpi=self.dpi,  bbox_inches='tight')

    def reset_zoom_location(self):
        sub_vec_x = self.vecs[1][self.zoom_sub_vec_num,0]
        sub_vec_y = self.vecs[1][self.zoom_sub_vec_num,1]
        x_width = abs(self.axins_kwargs['xlim'][1] - self.axins_kwargs['xlim'][0])
        self.axins_kwargs['xlim'] = (sub_vec_x-x_width/2, sub_vec_x+x_width/2)
        y_width = abs(self.axins_kwargs['ylim'][1] - self.axins_kwargs['ylim'][0])
        self.axins_kwargs['ylim'] = (sub_vec_y-y_width/2, sub_vec_y+y_width/2)

    def get_vec_colors(self):
        """Get colors for each vector"""
        sub_vec_colors = np.repeat(self.def_colors, self.vec_nums[1]) # repeat colors for each sub vector
        sub_vec_colors = sub_vec_colors[:self.vec_nums[1]*self.vec_nums[0]] # trim to number of vectors
        self.vec_colors = [sub_vec_colors]

        sub_sub_vec_colors = np.repeat(self.def_colors, self.vec_nums[1]*self.vec_nums[2]) # repeat colors for each sub vector
        sub_sub_vec_colors = sub_sub_vec_colors[:self.vec_nums[2]*self.vec_nums[1]*self.vec_nums[0]] # trim to number of vectors
        self.vec_colors.append(sub_sub_vec_colors)

        if self.vec_nums[3] > 0:
            sss_vec_colors = np.repeat(self.def_colors, self.vec_nums[1]*self.vec_nums[2]*self.vec_nums[3])
            sss_vec_colors = sss_vec_colors[:self.vec_nums[3]*self.vec_nums[2]*self.vec_nums[1]*self.vec_nums[0]] # trim to number of vectors
            self.vec_colors.append(sss_vec_colors)

    def scatter_vecs_for(self, ax, marker_multiplier=1):
        """Produce scatter plot"""
        for i, modality in enumerate(self.modalities):
            ax.scatter(self.vecs[0][i,0], self.vecs[0][i,1], s=self.marker_sizes[0]*marker_multiplier, 
                color=self.def_colors[i])
            if modality in ['SEP', 'CLS','MASK']:
                continue # skip sub vectors for these modalities
            id0, id1 = i*self.vec_nums[1], (i+1)*self.vec_nums[1]
            colors = self.vec_colors[0]
            if modality == 'Lab Test':
                # multiply vectors by random number
                multiplier = self.rng.uniform(0.5, .9, size=id1-id0)
                x_origins = np.repeat(self.vecs[0][i,0], id1-id0)
                y_origins = np.repeat(self.vecs[0][i,1], id1-id0)
                xy = np.vstack([x_origins, y_origins])
                U = self.get_vec(id1-id0, self.vector_lens[1])
                U = U*multiplier[:,None]
                ax.quiver(*xy, U[:,0], U[:,1],
                    color=colors[id0:id1], width=.001)
                continue
            else:
                ax.scatter(self.vecs[1][id0:id1,0], self.vecs[1][id0:id1,1], 
                        s=self.marker_sizes[1]*marker_multiplier, color=colors[id0:id1])
            
            id0, id1 = i*self.vec_nums[1]*self.vec_nums[2], (i+1)*self.vec_nums[1]*self.vec_nums[2]
            colors = self.vec_colors[1]
            ax.scatter(self.vecs[2][id0:id1,0], self.vecs[2][id0:id1,1], s=self.marker_sizes[2]*marker_multiplier, 
                color=colors[id0:id1])
            if self.vec_nums[3] > 0:
                id0, id1 = i*self.vec_nums[1]*self.vec_nums[2]*self.vec_nums[3], (i+1)*self.vec_nums[1]*self.vec_nums[2]*self.vec_nums[3]
                colors = self.vec_colors[2]
                ax.scatter(self.vecs[3][id0:id1,0], self.vecs[3][id0:id1,1], s=self.marker_sizes[3]*marker_multiplier, 
                    color=colors[id0:id1])

    def get_vector_hierarchy(self):
        """Create cluster of vectors for each modality, then create sub vectors for each cluster, then sub-sub vectors, etc.
        Returns list where each entry is an array of vectors for that level of hierarchy."""
        vectors = self.get_vec(self.vec_nums[0],self.vector_lens[0])
        sub_vec = []
        sub_sub_vec = []
        sss_vec = []
        for i in range(self.vec_nums[0]):
            sub_vec.append(vectors[i] + self.get_vec(self.vec_nums[1], self.vector_lens[1]))
            for j in range(self.vec_nums[1]):
                sub_sub_vec.append(sub_vec[i][j] + self.get_vec(self.vec_nums[2],self.vector_lens[2]))
       
        sub_vec = np.concatenate(sub_vec, axis=0)
        sub_sub_vec = np.concatenate(sub_sub_vec, axis=0)
        if self.vec_nums[3] > 0:
            for k in range(len(sub_sub_vec)):
                sss_vec.append(sub_sub_vec[k] + self.get_vec(self.vec_nums[3], self.vector_lens[3]))
        if self.vec_nums[3] > 0:
            sss_vec = np.concatenate(sss_vec, axis=0)
            return [vectors, sub_vec, sub_sub_vec, sss_vec]
        else:
            return [vectors, sub_vec, sub_sub_vec]
            
    def get_vec(self, n, len):
        """Get n vectors of length len"""
        vec = self.rng.uniform(-1,1,(n,2))
        vec = len*vec/la.norm(vec, axis=1).reshape(-1,1)
        return vec
    
    def get_nice_colors_dark_background(self):
        """Get nice additional colors for dark and light backgrounds"""
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        if self.style == 'dark_background':
            add_colors = ['#45bcde', '#52d273', '#e94f64', '#e57254']
        else:
            add_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        for color in add_colors:
            colors.insert(0, color)
        return  colors
    def turn_off_frame(self):
        """Turn off frame of plot"""
        self.ax.axes.get_xaxis().set_visible(False)
        self.ax.axes.get_yaxis().set_visible(False)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['bottom'].set_visible(False)
        self.ax.spines['left'].set_visible(False)
        return self.ax