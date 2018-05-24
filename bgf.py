"""
Code to model large galaxies from images
Martin Landriau, LBNL
Kaitlin DeVries
2017-18
"""
from __future__ import print_function
from __future__ import division
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import sys
from numpy import rec
from astropy.modeling.models import Sersic2D
from scipy.optimize import least_squares
from astropy.io import ascii
from mpl_toolkits.mplot3d import axes3d
from urllib.request import urlopen

class sersic_fit:

    def __init__(self, brickname, band, centers, subimage=None, srcdir=None):
        if srcdir==None:
            srcdir= "./"+brickname
        self.brickname = brickname
        self.band = band
        imagename = srcdir+'/legacysurvey-'+brickname+'-image-'+band+'.fits'
        image0 = fits.getdata(imagename)
        invarname = srcdir+'/legacysurvey-'+brickname+'-invvar-'+band+'.fits'
        invar0 = fits.getdata(invarname)
        chiname = srcdir+'/legacysurvey-'+brickname+'-chi2-'+band+'.fits'
        chi20 = fits.getdata(chiname)
        self.ngal = len(centers)
        x1, y1 = centers[0]
        if self.ngal == 2:
            x2, y2 = centers[1]
        else:
            x2 = 0
            y2 = 0
        if subimage==None:
            self.image = image0
            self.invar = invar0
            self.chi2_tractor = chi20
            self.xs = 0
            self.ys = 0
            self.xe = 0
            self.ye = 0
        else:
            xs, ys, xe, ye = subimage
            self.image = image0[xs:xe,ys:ye]
            self.invar = invar0[xs:xe,ys:ye]
            self.chi2_tractor = chi20[xs:xe,ys:ye]
            self.xs = xs
            self.ys = ys
            self.xe = xe
            self.ye = ye
        self.invar[self.invar<0.0] = 0.0
        self.n1, self.n2 = self.image.shape
        self.x, self.y = np.meshgrid(np.arange(self.n1), np.arange(self.n2), indexing='ij') # Vertical / horizontal axes convention?
        self.y1 = y1
        self.y2 = y2
        self.x1 = x1
        self.x2 = x2

    def get_model(self, params):    
        r = params[0]
        e = params[1]
        t = params[2]
        amp = params[3]
        model = Sersic2D(r_eff = r, n=4, x_0=self.x1-self.xs, y_0=self.y1-self.ys, ellip=e, theta=t, amplitude=amp)
        img = model(self.x,self.y)
        if len(params)==8:
            r2 = params[4]
            e2 = params[5]
            t2 = params[6]
            amp2 = params[7]
            model2 = Sersic2D(r_eff = r2, n=4, x_0=self.x2-self.xs, y_0=self.y2-self.ys, ellip=e2, theta=t2, amplitude=amp2)
            img2 = model2(self.x,self.y)
            img = img + img2
        resid = np.sqrt(self.invar)*(self.image-img)
        return img, resid

    def plot_model_1d_row(self, img, chi2, row=None):
        if row==None:
            if self.ngal == 1:
                row = int(self.x1)
            elif self.ngal == 2:
                row = (int(self.x2)+int(self.x1))//2
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        ax1.set_title("Brick " + self.brickname + " " + self.band + "-band image, row " +str(row))
        ii = np.where(self.invar>0.0)
        error = np.zeros((self.n1, self.n2))
        error[ii] = 1.0/np.sqrt(self.invar[ii])
        xaxis = np.arange(self.ys, self.ys+self.n2)
        row = row - self.xs
        line_data = ax1.semilogy(xaxis, self.image[row,0:self.n2], label='Data', lw=1)
        line_model = ax1.semilogy(xaxis, img[row,0:self.n2], label='Model', lw=1)
        ax1.set_ylabel('Flux')
        ax1.legend(handles=[line_data[0], line_model[0]])
        line_chi2 = ax2.semilogy(xaxis, chi2[row,0:self.n2], label='$\chi^2$, this code', lw=1)
        line_chi2tractor = ax2.semilogy(xaxis, self.chi2_tractor[row,0:self.n2], label='$\chi^2$, tractor', lw=1)
        ax2.set_ylabel('$\chi^2$')
        ax2.legend(handles=[line_chi2tractor[0], line_chi2[0]])
        plt.show()
    
    def plot_model_1d_column(self, img, chi2, column=None):
        if column==None:
            if self.ngal == 1:
                column = int(self.y1)
            elif self.ngal == 2:
                column = (int(self.y2)+int(self.y1))//2
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        ax1.set_title("Brick " + self.brickname + " " + self.band + "-band image, column " +str(column))
        ii = np.where(self.invar>0.0)
        error = np.zeros((self.n1, self.n2))
        error[ii] = 1.0/np.sqrt(self.invar[ii])
        xaxis = np.arange(self.xs, self.xs+self.n1)
        column -= self.ys
        line_data = ax1.semilogy(xaxis, self.image[0:self.n1,column], label='Data', lw=1)
        line_model = ax1.semilogy(xaxis, img[0:self.n1,column], label='Model', lw=1)
        ax1.set_ylabel('Flux')
        ax1.legend(handles=[line_data[0], line_model[0]])
        line_chi2 = ax2.semilogy(xaxis, chi2[0:self.n1,column], label='$\chi^2$, this code', lw=1)
        line_chi2tractor = ax2.semilogy(xaxis, self.chi2_tractor[0:self.n1,column], label='$\chi^2$, tractor', lw=1)
        ax2.set_ylabel('$\chi^2$')
        ax2.legend(handles=[line_chi2tractor[0], line_chi2[0]])
        plt.show()
    
    def model3d(self, img):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        X = self.x
        Y = self.y
        Z = np.log10(img)
        ax.plot_wireframe(X, Y, Z, rstride=7, cstride=7)
        plt.title("2D Model of Brick " + self.brickname + " " + self.band + "-band")
        plt.show()
    
    def get_residuals(self, p):
        img, resid = self.get_model(p)
        ix1 = int(self.x1-self.xs)
        iy1 = int(self.y1-self.ys)
        resid[ix1-1:ix1+2, iy1-1:iy1+2] = 0.0
        if self.ngal == 2:
            ix2 = int(self.x2-self.xs)
            iy2 = int(self.y2-self.ys)
            resid[ix2-1:ix2+2, iy2-1:iy2+2] = 0.0
        return np.ravel(resid)

    def best_fit(self, brickname, band, params, centers, srcdir=None):
        self.brickname = brickname
        self.band = band
        x1, y1 = centers[0]
        if self.ngal == 2:
            x2, y2 = centers[1]
        else:
            x2 = 0
            y2 = 0
        if srcdir==None:
            srcdir= "./"+brickname
        p = np.asarray(params)
        if self.ngal == 1:
            upper = np.array([np.inf, 1, 180, np.inf])
            results = least_squares(self.get_residuals, p, bounds=(0.0, upper))
        else:
            upper =  (np.inf, 1, np.pi, np.inf, np.inf, 1, np.pi, np.inf)
            bounds = (0.0, upper)
            results = least_squares(self.get_residuals, p, bounds=bounds)
        popt = results.x
        sigma = np.std(results.fun)
        H = np.dot(np.transpose(results.jac), results.jac)
        cov = (np.linalg.inv(H))*(sigma**2)
        errors = np.sqrt(np.diag(cov))
        popt = np.asarray(popt)
        data, header = fits.getdata(srcdir+'/legacysurvey-'+brickname+'-image-'+band+'.fits', header=True)
        hdu_number = 0
        fits.getheader(srcdir+'/legacysurvey-'+brickname+'-image-'+band+'.fits', hdu_number)
        CRVAL1 = header['CRVAL1']
        CRPIX1 = header['CRPIX1']
        CD1_1 = header['CD1_1']
        naxis1 = header['NAXIS1']
        naxis2 = header['NAXIS2']
        i = y1
        j = x1
        CRVAL2 = header['CRVAL2']
        CRPIX2 = header['CRPIX2']
        CD2_2 = header['CD2_2']
        ra = CRVAL1 + (i-CRPIX1)*(CD1_1)
        dec = CRVAL2 + (j-CRPIX2)*(CD2_2)
        if self.ngal == 2:
            i2 = y2
            ra2 = CRVAL1 + (i2-CRPIX1)*(CD1_1)
            j2 = x2
            dec2 = CRVAL2 + (j2-CRPIX2)*(CD2_2)
        decd = int(dec)
        decm = abs(int((dec-decd)*60))
        decs = (abs((dec-decd)*60)-decm)*60
        rah = int(ra/15)
        ram = int(((ra/15)-rah)*60)
        ras = int(((((ra/15)-rah)*60)-ram)*60)
        radius = 4.17
        web = "http://leda.univ-lyon1.fr/fG.cgi?n=a000&c=o&of=1,leda,ned&p=J%20"+str(rah)+"%20h%20"+str(ram)+"%20m%20"+str(ras)+"%20s,%20%20%20"+str(decd)+"d%20"+str(decm)+"m%20"+str(decs)+"s&f=box["+str(radius)+"]&ob=ra&nra=l&fql=[objtype%20=%27G%27],[type%20IN(%27E-S0%27,%20%27E%27,%20%27S0%27,%20%27S0-a%27)]&a=t&z=d"
        data = urlopen(web).read().decode('utf-8').split("\n")
        print("")
        galaxies = []
        for line in data:
            if len(line)>0:
                if line[0] != '#':
                    galchar = line[:-1].split()
                    first = galchar[0]
                    if first[0] in {'N', 'U', 'I'}:
                        galaxies.append(galchar[0])
                else:
                    pass       
        if self.ngal == 1:
            galaxy1name = galaxies
            a1 = np.array([galaxy1name])
            a2 = np.array([ra])
            a3 = np.array([dec])
            a4 = np.array([popt[0]])
            a4e = np.array([errors[0]])
            a5 = np.array([popt[1]])
            a5e = np.array([errors[1]])
            a6 = np.array([4.0])
            a7 = np.array([brickname])
            a8 = np.array([band])
        elif self.ngal == 2:
            galaxy1name, galaxy2name = galaxies
            a1 = np.array([galaxy1name, galaxy2name])
            a2 = np.array([ra, ra2])
            a3 = np.array([dec, dec2])
            a4 = np.array([popt[0], popt[4]])
            a4e = np.array([errors[0], errors[4]])
            a5 = np.array([popt[1], popt[5]])
            a5e = np.array([errors[1], errors[5]])
            a6 = np.array([4.0, 4.0])
            a7 = np.array([brickname, brickname])
            a8 = np.array([band, band])
        #print(a1)
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
        tbhdu = fits.BinTableHDU.from_columns(cols)
        tbhdu.writeto('big.fits', overwrite= True)
        file = fits.getdata('big.fits')
        print('\n')
        print(file.names)
        print(file)
        """filea = fits.getdata('big.fits')
        t1 = fits.open('this.fits')
        t2 = fits.open('big.fits')
        nrows1 = t1[1].data.shape[0]
        nrows2 = t2[1].data.shape[0]
        nrows = nrows1 + nrows2
        hdu = fits.BinTableHDU.from_columns(t1[1].columns, nrows=nrows)
        for colname in t1[1].columns.names:
            hdu.data[colname][nrows1:] = t2[1].data[colname]
        hdu.writeto('big.fits', overwrite= True)
        filet = fits.getdata('big.fits')"""
        print ('\033[1m'+ '\nBest fit parameters' + '\033[0m')
        print ('Number of iterations= ', results.nfev)
        print ('For Galaxy #1:')
        print ('(RA, DEC) =', '(', ra, ',', dec, ')')
        print ('\teffective radius =', popt[0], " +/- ", errors[0])
        print ('\tellipticity = ', popt[1], " +/- ", errors[1])
        print ('\ttheta= ', np.degrees(popt[2]), " +/- ", np.degrees(errors[2]))
        print ('\tamplitude= ', popt[3], " +/- ", errors[3])
        if self.ngal==2:
            print ('For Galaxy #2:')
            print ('(RA, DEC) =', '(', ra2, ',', dec2, ')')
            print ('\teffective radius =', popt[4], " +/- ", errors[4])
            print ('\tellipticity = ', popt[5], " +/- ", errors[5])
            print ('\ttheta= ', np.degrees(popt[6]), " +/- ", np.degrees(errors[6]))
            print ('\tamplitude= ', popt[7]," +/- ", errors[7])
        return popt


# Main code
def main(brickname, band):

    srcdir = brickname

    image = fits.getdata(srcdir+'/legacysurvey-'+brickname+'-image-'+band+'.fits')
    n1, n2 = image.shape
    plt.imshow(image, vmin=-.1, vmax=.1, cmap='BuPu_r')
    plt.colorbar()
    plt.title("Brick " + brickname + " " + band + "-band image")

    # This is currently interactive, but should be the blob size
    print("\nChoose subregion (upper left,lower right)")
    plt.show()

    ys, xs, ye, xe = input("Subregion: ").split(",")
    xs = int(xs)
    ys = int(ys)
    xe = int(xe)
    ye = int(ye)
    subimage = image[xs:xe,ys:ye]

    neighbourhood_size = 10
    threshold = 0.1

    data_max = filters.maximum_filter(subimage, neighbourhood_size)
    maxima = (subimage == data_max)
    data_min = filters.minimum_filter(subimage, neighbourhood_size)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0

    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)
    x, y = [], []
    for dy,dx in slices:
        x_center = (dx.start + dx.stop - 1)/2
        x.append(x_center+ys)
        y_center = (dy.start + dy.stop - 1)/2
        y.append(y_center+xs)

    print("\nLocation of maxima:")
    nmax = len(x)
    for i in range(nmax):
        print(str(i)+" "+str(x[i])+" "+str(y[i]))

    print('\nChoose the galaxies to be fit. x0 and y0 should be taken from the list above.')
    plt.imshow(subimage, vmin=-.1, vmax=.1, extent=(ys,ye,xe,xs), cmap='BuPu_r')
    plt.colorbar()
    plt.title("Brick " + brickname + " " + band + "-band image")
    plt.plot(x,y, 'bo')
    plt.show()

    igal = int(input("\nChoose first galaxy from the list. "))
    twoGalaxies=True
    if (igal>=0 and igal<nmax):
        x1 = y[igal]
        y1 = x[igal]
    else:
        sys.exit("Galaxy must be between 0 and "+str(nmax-1)+".")
    igal = int(input("Choose second galaxy from the list. Enter -1 to fit only one galaxy. "))
    if igal == -1:
        twoGalaxies = False
    elif (igal>=0 and igal<nmax):
        x2 = y[igal]
        y2 = x[igal]
    else:
        sys.exit("Galaxy must be between 0 and "+str(nmax-1)+".")
    plt.show(block=True)

    a = 200
    b = 100
    ti = 90
    t = ti*(np.pi)/180
    r = 135
    e = .5
    amp = .09

    if twoGalaxies:
        a2 = 200
        b2 = 100
        ti2 = 90
        t2 = ti2*(np.pi)/180
        r2 = 60
        e2 = .2
        amp2 = .5
        centers = [(x1,y1),(x2,y2)]
        params = (r, e, t, amp, r2, e2, t2, amp2)
    else:
        centers = [(x1,y1)]
        params = (r, e, t, amp)

    brick = sersic_fit(brickname, band, centers, subimage=(xs,ys,xe,ye))

    p = brick.best_fit(brickname, band, params, centers)
    img, resid = brick.get_model(p)
    brick.plot_model_1d_row(img, resid**2)

    answer = input('\nView 2D plot? (Y/N): ')
    if answer == "Y" or answer == "y":
        brick.model3d(img)
        #sys.exit()
    #else:
        #sys.exit()
    sys.exit()


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("brickname", type=str, help="Brick name")
    parser.add_argument("band", type=str, help="Band")
    args = parser.parse_args()
    main(args.brickname, args.band)

