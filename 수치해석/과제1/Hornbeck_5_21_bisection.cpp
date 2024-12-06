#include <stdio.h>
#include <math.h>

#define func(x) x*x*x*x - 7.223*x*x*x + 13.447*x*x - 0.672*x -10.223

main ()
{
	
	FILE *out1; // 파일 변수 선언
	out1=fopen("Hornbeck_5_21_bisection.dat","w"); 
	// 디렉토리와 디렉토리 내에서 출력할 파일 지정

    int imax,iter;
    double a,b,xc,fa,fb,fc,epsil;
    
    epsil = 0.00000001;
    imax=10000;
    
    a= -1000.0;
	b= 0.0;
	
	fa=func(a);
	fb=func(b);
	
	if(fabs(fa)<=epsil) 
	{
	   fprintf(out1,"root x = %f \n", a);
	}

    if(fabs(fb)<=epsil) 
	{
	   fprintf(out1,"root x = %f \n", b);
	}

    if(fa*fb>0.0) 
	{
	   fprintf(out1,"no root between a and b  \n");
	   goto END;
	}


    for(iter=1;iter<imax;++iter)
       {
       	xc=(a+b)/2.0;
       	fc=func(xc);
       	
       	 fprintf(out1,"a = %f, b = %f \n", a, b);
       	
       	if(fabs(fc)<=epsil)
       	{
       		fprintf(out1,"root x = %f \n", xc);
       		goto END;
		   }
		if(fa*fc<0.0)
		{
		    b=xc;
			fb=fc;	
		}
		else
		{
		    a=xc;
			fa=fc;	
		}
		
	   }
	   
	END:;
	
	fclose(out1);
}
