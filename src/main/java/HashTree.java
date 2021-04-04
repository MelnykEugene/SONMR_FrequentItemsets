
import java.util.ArrayList;
import java.util.List;

public class HashTree {
    private int h=30;
    private int itemset_size;
    int population;
    InnerNode root;
    LeafNode last_entry=null;

    public HashTree(int itemset_size, int h){
        this.h=h;
        this.itemset_size=itemset_size;
        root= new InnerNode();
    }

    abstract class Node{}
    class LeafNode extends  Node{

    }
    class InnerNode extends Node{

    }

}

