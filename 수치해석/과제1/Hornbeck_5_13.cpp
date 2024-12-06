#include <stdio.h>
#include <math.h>


#define func(x) x - sqrt(3)
#define func_diff(x) 1

main ()
{
	
	FILE *out1;
	out1=fopen("Hornbeck_5_13_result.dat","w");

    int imax,iter;
    double x,fx,fdx,del,epsil;
    
    epsil = 0.1;
    imax=100;
    
    x=2;
	
	
	fx=func(x);
	fdx=func_diff(x);
	


    for(iter=1;iter<imax;++iter)
       {
       	
		del=-fx/fdx; 
		x=x+del;
		
		 fprintf(out1,"x = %f, del = %f \n", x, del);
		
		if(fabs(del)<epsil)
		{
		     fprintf(out1,"root x = %f \n", x);
			 goto END;	
		}
		
		fx=func(x);
		fdx=func_diff(x);
		
       
	         	
	   }
	   
	END:;
	
	fclose(out1);
}
