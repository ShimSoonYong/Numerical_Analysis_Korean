#include <stdio.h>
#include <math.h>

#define func(x) x*x*x*x - 7.223*x*x*x + 13.447*x*x - 0.672*x -10.223

main ()
{
	
	FILE *out1;
	out1=fopen("Hornbeck_5_21_result.dat","w");

    int imax,iter;
    double x,fx,xo,fxo,del,epsil;
    
    epsil = 0.000001;
    imax=1000;
    
    xo=1000; 
	x= 1000.1;
	del=x-xo;
	
	
	fxo=func(xo);
	fx=func(x);
	
	


    for(iter=1;iter<imax;++iter)
       {
       	
      /* 	fprintf(out1,"fxo = %f , fx = %f \n", fxo,fx);   */
       	
		del=-fx/((fx-fxo)/del); 
		x=x+del;
		
		fprintf(out1,"x = %f, del = %f \n", x, del);
		
		if(fabs(del)<epsil)
		{
		     fprintf(out1,"root x = %f \n", x);
			 goto END;	
		}
		
		fxo=fx;
		fx=func(x);
		
				       	
	   }
	   
	END:;
	
	fclose(out1);
}
