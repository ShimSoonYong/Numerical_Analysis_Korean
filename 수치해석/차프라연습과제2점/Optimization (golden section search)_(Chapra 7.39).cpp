#include <stdio.h>
#include <math.h>

#define func(x) (4/cos(x) + 4/sin(x))

main ()
{
	
	FILE *out1;
	out1=fopen("extreme value (golden section search)_(Chapra 7.39).dat","w");

    int imax,iter;
    double xa,xb,xc,xd,xm,l_golden,fxa,fxb,fxc,fxd,fxm;
    double k_golden,epsil,res;
    
    epsil = 0.000000000001;
	imax=1000;
    k_golden=(sqrt(5.0)+1.0)/(sqrt(5.0)+3.0);
    
    xa= 0.0;
	xb= M_PI/2;
	
	fxa=func(xa);
	fxb=func(xb);
	
    for(iter=1;iter<imax;++iter)
       {
       	l_golden=k_golden*fabs(xb-xa);
       	
       	xd=xa+l_golden;    /*   x1   */
       	xc=xb-l_golden;    /*   x2   */
       	xm=(xd+xc)/2.0;
       	
       	fxd=func(xd);
       	fxc=func(xc);
       	fxm=(fxd+fxc)/2.0;
       	
       	res=fabs(fxd-fxc);
       	
       	fprintf(out1,"#iter%d, xm = %f, res = %f, extreme = %f \n"
		            ,iter,xm,res,fxm);
       	       	      	
       	if(res<epsil)
       	{
       		fprintf(out1," \n\n");
			fprintf(out1,"xm = %f, extreme value = %f \n", xm,fxm);
			goto END;
       		
		}
				          	
       	if((fxc)<(fxd)) /* high --> fxc > fxd and low --> fxc < fxd */       	                
       	{
            xb=xd;
		}
		
		else
		{
		    xa=xc;
		}
				
	   }
	   
	END:;
		
	
	fclose(out1);
}
