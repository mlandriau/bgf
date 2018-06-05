from __future__ import print_function
from __future__ import division
from astropy.io import fits
from astropy import wcs
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import sys
from numpy import rec
import os
import pdb
from astropy.modeling.models import Sersic2D
from scipy.optimize import least_squares
from astropy.io import ascii
from mpl_toolkits.mplot3d import axes3d
from urllib.request import urlopen

class sersic_fit:

	def __init__(self, brickname, band, centers=None, subimage=None, srcdir=None):
		if srcdir==None:
			srcdir= "./"+brickname
		self.brickname = brickname
		self.band = band
		imagename = os.path.join(srcdir, 'legacysurvey-%s-image-%s.fits.fz' % (brickname, band))
		image0 = fits.getdata(imagename)
		invarname = os.path.join(srcdir, 'legacysurvey-%s-invvar-%s.fits.fz' % (brickname, band))
		invar0 = fits.getdata(invarname)
		chiname = os.path.join(srcdir, 'legacysurvey-%s-chi2-%s.fits.fz' % (brickname, band))
		chi20 = fits.getdata(chiname)
		xs, ys, xe, ye = subimage
		xs = int(xs)
		ys = int(ys)
		xe = int(xe)
		ye = int(ye)
		self.image = image0[ye:ys,xs:xe]
		self.invar = invar0[ye:ys,xs:xe]
		self.chi2_tractor = chi20[ye:ys,xs:xe]
		# plt.imshow(self.image, vmin=-.001, vmax=1, cmap="pink")
# 		plt.title("Image")
# 		plt.show()
# 		plt.imshow(self.invar, vmin=-.001, vmax=1, cmap="pink")
# 		plt.title("Invar")
# 		plt.show()
# 		plt.imshow(self.chi2_tractor, vmin=-.001, vmax=1, cmap="pink")
# 		plt.title("Chi2 Tractor")
# 		plt.show()
		self.xs = xs
		self.ys = ys
		self.xe = xe
		self.ye = ye
		x1 = centers[0]
		y1 = centers[1]
		self.invar[self.invar<0.0] = 0.0
		self.n1, self.n2 = self.image.shape
		self.x, self.y = np.meshgrid(np.arange(self.n1), np.arange(self.n2), indexing='ij') # Vertical / horizontal axes convention?
		self.y1 = y1
		self.x1 = x1

	def get_model(self, params):    
		r = params[0]
		e = params[1]
		t = params[2]
		amp = params[3]
		model = Sersic2D(r_eff = r, n=4, x_0=self.x1-self.xs, y_0=self.y1-self.ye, ellip=e, theta=t, amplitude=amp)
		img = model(self.x,self.y)
		resid = np.sqrt(self.invar)*(self.image-img)
		return img, resid

	def plot_model_1d_row(self, img, chi2, row=None):
		if row==None:
			row = int(self.y1)
		fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
		ax1.set_title("Brick " + self.brickname + " " + self.band + "-band image, row " +str(row))
		ii = np.where(self.invar>0.0)
		error = np.zeros((self.n1, self.n2))
		error[ii] = 1.0/np.sqrt(self.invar[ii])
		xaxis = np.arange(self.xs, self.xs+self.n1)
		row = row - self.ye
		line_data = ax1.semilogy(xaxis, self.image[0:self.n1, row], label='Data', lw=1)
		line_model = ax1.semilogy(xaxis, img[0:self.n1, row], label='Model', lw=1)
		ax1.set_ylabel('Flux')
		ax1.legend(handles=[line_data[0], line_model[0]])
		line_chi2 = ax2.semilogy(xaxis, chi2[0:self.n1, row], label='$\chi^2$, this code', lw=1)
		line_chi2tractor = ax2.semilogy(xaxis, self.chi2_tractor[0:self.n1, row], label='$\chi^2$, tractor', lw=1)
		ax2.set_ylabel('$\chi^2$')
		ax2.legend(handles=[line_chi2tractor[0], line_chi2[0]])
		plt.show()

	def get_residuals(self, p):
		img, resid = self.get_model(p)
		ix1 = int(self.x1-self.xs)
		iy1 = int(self.y1-self.ye)
		resid[ix1-1:ix1+2, iy1-1:iy1+2] = 0.0
		return np.ravel(resid)

	def best_fit(self, brickname, band, params, centers, srcdir=None):
		self.brickname = brickname 
		self.band = band
		x1 = centers[0]
		y1 = centers[1]
		if srcdir==None:
			srcdir= "./"+brickname
		p = np.asarray(params)
		upper = np.array([np.inf, 1, 180, np.inf])
		results = least_squares(self.get_residuals, p, bounds=(0.0, upper))
		popt = results.x
		sigma = np.std(results.fun)
		H = np.dot(np.transpose(results.jac), results.jac)
		cov = (np.linalg.inv(H))*(sigma**2)
		errors = np.sqrt(np.diag(cov))
		popt = np.asarray(popt)
		image = os.path.join(srcdir, 'legacysurvey-%s-image-%s.fits.fz' % (brickname, band))
		data, header = fits.getdata(image, header=True)
		fits.getheader(image, 0)
		twcs = wcs.WCS(header)
		ra, dec = twcs.wcs_pix2world(y1, x1, 0)
		decd = int(dec)
		decm = int(abs(int((dec-decd)*60)))
		decs = int((abs((dec-decd)*60)-decm)*60)
		rah = int(ra/15)
		ram = int(((ra/15)-rah)*60)
		ras = int(((((ra/15)-rah)*60)-ram)*60)
		radius = 4.17
		web = "http://leda.univ-lyon1.fr/fG.cgi?n=a000&c=o&of=1,leda,ned&p="+str(rah)+"h%2010%20m%20"+str(ram)+"%20s%20"+str(ras)+"%20d%20"+str(decd)+"%20m%20"+str(decm)+"%20s&f=4%2E"+str(decs)+"&ob=ra&nra=l&a=t&z=d"
		webData = urlopen(web).read().decode('utf-8').split("\n")
		print("")
		galaxies = []
		for line in webData:
			if len(line)>0:
				if line[0] != '#':
					galchar = line[:-1].split()
					first = galchar[0]
					if first[0] in {'N', 'U', 'I', 'P'}:
						galaxies.append(galchar[0])
				else:
					pass    
		galaxyname = galaxies[0]
		a1 = np.array(galaxyname)
		a2 = np.array([ra])
		a3 = np.array([dec])
		a4 = np.array([popt[0]])
		a4e = np.array([errors[0]])
		a5 = np.array([popt[1]])
		a5e = np.array([errors[1]])
		a6 = np.array([4.0])
		a7 = np.array([brickname])
		a8 = np.array([band])
		col1 = fits.Column(name='Name', format='100A', array=a1)
		col2 = fits.Column(name='RA', format='E', array=a2)
		col3 = fits.Column(name='Dec', format='E', array=a3)
		col4 = fits.Column(name='Effective radius', format='E', array=a4)
		col4e = fits.Column(name='Margin of Error for Eff. Radius', format='E', array=a4e)
		col5 = fits.Column(name='Ellipticity', format='E', array=a5)
		col5e = fits.Column(name='Margin of Error for Ellip.', format='E', array=a5e)
		col6 = fits.Column(name='N', format='A', array=a6)
		col7 = fits.Column(name='Brick', format='8A', array=a7)
		col8 = fits.Column(name='Band', format='A', array=a8)
		cols = fits.ColDefs([col1, col2, col3, col4, col4e, col5, col5e, col6, col7, col8]) 
		"""tbhdu = fits.BinTableHDU.from_columns(cols)
		tbhdu.writeto('big.fits', overwrite= True)
		file = fits.getdata('big.fits')
		print('\n')
		print(file.names)
		print(file)
		filea = fits.getdata('big.fits')
		t1 = fits.open('this.fits')
		t2 = fits.open('big.fits')
		nrows1 = t1[1].data.shape[0]
		nrows2 = t2[1].data.shape[0]
		nrows = nrows1 + nrows2
		hdu = fits.BinTableHDU.from_columns(t1[1].columns, nrows=nrows)
		for colname in t1[1].columns.names:
			hdu.data[colname][nrows1:] = t2[1].data[colname]
		hdu.writeto('big.fits', overwrite= True)
		filet = fits.getdata('big.fits')
		"""
		print ('\033[1m'+ '\nBest fit parameters' + '\033[0m')
		print ('Number of iterations= ', results.nfev)
		print ('For Galaxy #1:')
		print ('(RA, DEC) =', '(', ra, ',', dec, ')')
		print ('\teffective radius =', popt[0], " +/- ", errors[0])
		print ('\tellipticity = ', popt[1], " +/- ", errors[1])
		print ('\ttheta= ', np.degrees(popt[2]), " +/- ", np.degrees(errors[2]))
		print ('\tamplitude= ', popt[3], " +/- ", errors[3])
		return popt

def main(brickname, band):
	srcdir = brickname
	imagename = os.path.join(srcdir, 'legacysurvey-%s-image-%s.fits.fz' % (brickname, band))
	im0 = fits.getdata(imagename)
	model = os.path.join(srcdir, 'all-models-%s.fits' % (brickname))
	hdu = fits.open(model)
	data = hdu[1].data
	large = np.where(data['blob_totalpix'] == np.max(data['blob_totalpix']))
	x = data[large]
	raLeft = np.max(x['psf_ra'])
	raRight = np.min(x['psf_ra'])
	decUp = np.max(x['psf_dec'])
	decDown = np.min(x['psf_dec'])
	header = hdu[0].header

	neighborhood_size = 1000
	threshold = 1

	im0_max = filters.maximum_filter(im0, neighborhood_size)
	maxima = (im0 == im0_max)
	im0_min = filters.minimum_filter(im0, neighborhood_size)
	diff = ((im0_max - im0_min) > threshold)
	maxima[diff == 0] = 0
	labeled, num_objects = ndimage.label(maxima)
	xy = np.array(ndimage.center_of_mass(im0, labeled, range(1, num_objects+1)))

	data, header = fits.getdata(imagename, header=True)
	twcs = wcs.WCS(header)
	xs, ys = twcs.wcs_world2pix(raLeft, decUp, 0)
	xe, ye = twcs.wcs_world2pix(raRight, decDown, 0)

	centers = []
	i = 0
	for i in range(0, len(xy[:, 0])):
		if xy[i, 1] > xs and xy[i, 1] < xe:
			if xy[i, 0] < ys and xy[i, 0] > ye:
				centers.append(xy[i, 1])
				centers.append(xy[i, 0])
		i = i + 1
	xCenter = centers[0]
	yCenter = centers[1]
	plt.imshow(im0, cmap="pink", vmin =-.001, vmax=.5)
	plt.title("Subimage %s, band %s" % (brickname, band))
	plt.plot(xCenter, yCenter, 'bo')
	plt.colorbar()
	#plt.show()

	a = 200
	b = 100
	ti = 90
	t = ti*(np.pi)/180
	r = 135
	e = .5
	amp = .09
	params = (r, e, t, amp)
	brick = sersic_fit(brickname, band, centers, subimage=(xs, ys, xe, ye))
	p = brick.best_fit(brickname, band, params, centers)
	img, resid = brick.get_model(p)
	brick.plot_model_1d_row(img, resid**2)
	
if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("brickname", type=str, help="Brick name")
    parser.add_argument("band", type=str, help="Band")
    args = parser.parse_args()
    main(args.brickname, args.band)
