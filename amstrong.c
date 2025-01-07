#include<stdio.h>
void main()
{
   int org, num=100, temp,     
   sum=0, ams;
   while(num<501)
   {
     org==num;
     while(num!=0)
     {
       temp=num%10;
       num=num/10;
       ams=sum+(temp*temp*temp);
       sum=ams;
      }
       if (org==sum)
     {
       printf("No. is Amstrong: %d",sum);
     }
       num= num+1;
     }
}