import java.util.ArrayList;
import java.util.List;

public class Nodo {
    String estado;
    Nodo padre;
    int profundidad;
    int costo;

    public Nodo(String estado, Nodo padre){
        this.estado = estado;
        this.padre = padre;
        if(padre == null){
            this.profundidad = 0;
            this.costo = 0;
        } else {
            this.profundidad = padre.profundidad + 1;
            this.costo = padre.costo + 1;
        }
    }


     public List<String> obtenerSucesores() {
        List<String> successors = new ArrayList<String>();

        switch (estado.indexOf(" ")) {
            case 0: {
                successors.add(estado.replace(estado.charAt(0), '*').replace(estado.charAt(1), estado.charAt(0)).replace('*', estado.charAt(1)));
                successors.add(estado.replace(estado.charAt(0), '*').replace(estado.charAt(3), estado.charAt(0)).replace('*', estado.charAt(3)));
                break;
            }
            case 1: {
                successors.add(estado.replace(estado.charAt(1), '*').replace(estado.charAt(0), estado.charAt(1)).replace('*', estado.charAt(0)));
                successors.add(estado.replace(estado.charAt(1), '*').replace(estado.charAt(2), estado.charAt(1)).replace('*', estado.charAt(2)));
                successors.add(estado.replace(estado.charAt(1), '*').replace(estado.charAt(4), estado.charAt(1)).replace('*', estado.charAt(4)));
                break;
            }
            case 2: {

                successors.add(estado.replace(estado.charAt(2), '*').replace(estado.charAt(1), estado.charAt(2)).replace('*', estado.charAt(1)));
                successors.add(estado.replace(estado.charAt(2), '*').replace(estado.charAt(5), estado.charAt(2)).replace('*', estado.charAt(5)));
                break;
            }
            case 3: {
                successors.add(estado.replace(estado.charAt(3), '*').replace(estado.charAt(0), estado.charAt(3)).replace('*', estado.charAt(0)));
                successors.add(estado.replace(estado.charAt(3), '*').replace(estado.charAt(4), estado.charAt(3)).replace('*', estado.charAt(4)));
                successors.add(estado.replace(estado.charAt(3), '*').replace(estado.charAt(6), estado.charAt(3)).replace('*', estado.charAt(6)));
                break;
            }
            case 4: {
                successors.add(estado.replace(estado.charAt(4), '*').replace(estado.charAt(1), estado.charAt(4)).replace('*', estado.charAt(1)));
                successors.add(estado.replace(estado.charAt(4), '*').replace(estado.charAt(3), estado.charAt(4)).replace('*', estado.charAt(3)));
                successors.add(estado.replace(estado.charAt(4), '*').replace(estado.charAt(5), estado.charAt(4)).replace('*', estado.charAt(5)));
                successors.add(estado.replace(estado.charAt(4), '*').replace(estado.charAt(7), estado.charAt(4)).replace('*', estado.charAt(7)));
                break;
            }
            case 5: {
                successors.add(estado.replace(estado.charAt(5), '*').replace(estado.charAt(2), estado.charAt(5)).replace('*', estado.charAt(2)));
                successors.add(estado.replace(estado.charAt(5), '*').replace(estado.charAt(4), estado.charAt(5)).replace('*', estado.charAt(4)));
                successors.add(estado.replace(estado.charAt(5), '*').replace(estado.charAt(8), estado.charAt(5)).replace('*', estado.charAt(8)));
                break;
            }
            case 6: {
                successors.add(estado.replace(estado.charAt(6), '*').replace(estado.charAt(3), estado.charAt(6)).replace('*', estado.charAt(3)));
                successors.add(estado.replace(estado.charAt(6), '*').replace(estado.charAt(7), estado.charAt(6)).replace('*', estado.charAt(7)));
                break;

            }
            case 7: {
                successors.add(estado.replace(estado.charAt(7), '*').replace(estado.charAt(4), estado.charAt(7)).replace('*', estado.charAt(4)));
                successors.add(estado.replace(estado.charAt(7), '*').replace(estado.charAt(6), estado.charAt(7)).replace('*', estado.charAt(6)));
                successors.add(estado.replace(estado.charAt(7), '*').replace(estado.charAt(8), estado.charAt(7)).replace('*', estado.charAt(8)));
                break;
            }
            case 8: {
                successors.add(estado.replace(estado.charAt(8), '*').replace(estado.charAt(5), estado.charAt(8)).replace('*', estado.charAt(5)));
                successors.add(estado.replace(estado.charAt(8), '*').replace(estado.charAt(7), estado.charAt(8)).replace('*', estado.charAt(7)));
                break;
            }
        }
        return successors;
    }

}

