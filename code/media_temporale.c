#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv){
    
    if(argc != 2){
        fprintf(stderr, "Errore di parametri sbagliato\n");
        exit(1);
    }

    FILE *fp;
    if((fp = fopen(argv[1], "r")) == NULL){
        fprintf(stderr, "File non trovato\n");
        exit(1);
    }

    char buf[20];
    int min, sec, msec, usec;
    int sum = 0, count = 0;
    while(fread(buf, 14, 1, fp) != 0){ //00:00:000:000\n
        sscanf(buf, "%02d:%02d:%03d:%03d", &min, &sec, &msec, &usec);
        count++;
        sum += (min * 60000000) + (sec * 1000000) + (msec * 1000) + usec;
    }

    sum = sum / count;
    printf("Tempo medio (su %d immagini) per immagine [Foreweard e Backward]: %d microsecondi\n", count, sum);


    return 0;
}