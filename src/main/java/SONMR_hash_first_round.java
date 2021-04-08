import java.io.*;
import java.net.URI;
import java.util.*;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import static org.apache.hadoop.mapreduce.lib.input.NLineInputFormat.setNumLinesPerSplit;

public class SONMR_hash_first_round {
    public static void main(String[] args) throws Exception {
        int dataset_size= Integer.parseInt(args[0]);
        int transactions_per_block= Integer.parseInt(args[1]);
        String min_supp= args[2];
        String corr_factor = args[3];

        double starting_time= System.currentTimeMillis();

        Configuration conf = new Configuration();
        conf.set("minsup", min_supp);
        conf.set("corr",corr_factor);
        Job job = Job.getInstance(conf, "first round");
        job.setJarByClass(SONMR.class);
        job.setMapperClass(SONMR.AprioriMapper.class);
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(IntWritable.class);
        //job.setCombinerClass(FilterPresentItemsetsReducer.class);
        job.setReducerClass(SONMR.FilterPresentItemsetsReducer.class);
        job.setInputFormatClass(MultiLineInputFormat.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(NullWritable.class);
        setNumLinesPerSplit(job, transactions_per_block);
        FileInputFormat.addInputPath(job, new Path(args[4]));
        FileOutputFormat.setOutputPath(job, new Path(args[5]));

        job.waitForCompletion(false);

        conf = new Configuration();
        conf.set("minsup", min_supp);
        conf.set("corr",corr_factor);
        job = Job.getInstance(conf, "second round");
        job.setJarByClass(SONMR.class);
        job.setMapperClass(SONMR.SecondMapper.class);
        job.setMapOutputKeyClass(Text.class); //second mapper takes locally frequent itemsets, which we saved as Text
        job.setMapOutputValueClass(IntWritable.class);
        //job.setCombinerClass(FilterPresentItemsetsReducer.class); //not thinking of combiner for now
        job.setReducerClass(SONMR.SecondReduce.class);
        //job.setInputFormatClass(MultiLineInputFormat.class); we are using default one-line per mapper input
        job.setOutputKeyClass(Text.class); //the key is an itemset, i.e Text
        job.setOutputValueClass(IntWritable.class); //the value is the support, IntWritable
        FileInputFormat.addInputPath(job, new Path(args[4])); //the input is dataset
        FileOutputFormat.setOutputPath(job, new Path(args[6])); //the output path is the output directory
        Path candidates_path = new Path(args[5]+ File.separator + "part-r-00000");
        job.addCacheFile(candidates_path.toUri());
        job.waitForCompletion(false);
        double ending_time= System.currentTimeMillis();
        System.err.println("time ellapsed: "+String.valueOf((ending_time-starting_time)/1000) + " seconds");
    }

    public static class HashAprioriMapper
            extends Mapper<Object, Text, Text, IntWritable>{
//====================================================================== Auxiliary classes
        private Text word = new Text();
        private final static IntWritable one = new IntWritable(1);
        private int branching_factor;
        //wrapper class allowing us to store supports locally at each itemset for us to stay sane
        class ItemSet{
            int[] itemset;
            int support=0;
            ItemSet(int[] itemset){
                this.itemset=itemset;
            }
        }

        abstract class Node{}

        class LeafNode extends Node{
            public ArrayList<ItemSet> itemsets;
            LeafNode next_leaf;

            LeafNode(){
                itemsets= new ArrayList<>();
            }
        }

        class InteriorNode extends Node{
            Node[] childs;
            InteriorNode(int h){
                childs= new Node[h];
            }
        }

        class HashTree{
            private final int h; //branching factor
            private final int itemset_size; //length of candidates in this tree AKA tree height
            private int population=0;
            public int leaf_nodes=0;
            public InteriorNode root;
            //I chain leaf nodes into a linked list so that I can access them without traversing top-to-bottom
            public LeafNode last_leaf=null;

            HashTree(int itemset_size,int h){
                this.itemset_size=itemset_size;
                this.h=h;
                root = new InteriorNode(h);
            }
            //insertion is done recursively. this is entry point
            public void insert_candidate(ItemSet item_set){
                this.population++;
                this.insert(this.root, item_set,0);
            }
            private void insert(Node current_node, ItemSet item_set, int k){
                //if we reached a leaf, we insert
                if(current_node instanceof LeafNode){
                    ((LeafNode) current_node).itemsets.add(item_set);
                }
                //else we are at interior node
                else{
                    //pick which branch to follow
                    //the hash function is just the numerical value of current item in itemset
                    int direction = item_set.itemset[k] % this.h;
                    Node next_node = ((InteriorNode)current_node).childs[direction];
                    //if there is no next node, we must insert either a leaf node or an interior
                    //based on the depth reached so far
                    if (next_node==null) {
                        if (k==item_set.itemset.length-2){ //this means the bottom has been reached
                            next_node = new LeafNode();
                            this.leaf_nodes+=1;
                            //manage the linked list of leaf nodes:
                            ((LeafNode)next_node).next_leaf=this.last_leaf;
                            this.last_leaf= ((LeafNode) next_node);
                        }
                        else{
                            next_node=new InteriorNode(this.h);
                        }
                        //advance the recursion
                        this.insert(next_node,item_set,k+1);
                    }
                }
            }
            public boolean lookup(int[] item_set){
                int k =0;
                Node current_node = this.root;
                //for the itemset to be in the tree, its leaf must be reached in a certain number of steps so we can iterate
                while (k<item_set.length-1){
                    int direction = item_set[k]% this.h;
                    Node next_node = ((InteriorNode)current_node).childs[direction];
                    if (next_node==null) return false; //if the corresponding branch doesn't exist
                    current_node=next_node;
                    k+=1;
                }
                assert(current_node instanceof LeafNode);
                //the itemsets in leaf nodes All share the first k-1 items by construction
                //therefore we can know if the itemset is in the leaf based Only on the last item
                ArrayList<Integer> last_items = new ArrayList<Integer>();
                for(ItemSet set : ((LeafNode) current_node).itemsets){
                    last_items.add(set.itemset[set.itemset.length-1]);
                }
                return list_contains(last_items,item_set[item_set.length-1]);
            }

            private boolean list_contains(List<Integer> list, int last){
                for(int item : list){
                    //System.err.println("comparing "+Arrays.toString(item) + " with "+ Arrays.toString(candidate));
                    if(last == item) return true;
                }
                return false;
            }

            //entry-point for recursive calculation of supports
            public void update_supports(int[] transaction){
                this.update(this.root,transaction,0, new int[]{});
            }

            //http://www.cs.uoi.gr/~tsap/teaching/2012f-cs059/material/datamining-lect3.pdf
            //the link kinda explains what is going on (slides 43-44).
            //fixed is the prefix that's being explored by the current recursion branch
            private void update(Node node, int[] transaction, int first_position, int[] fixed){
                int last_possible_path= transaction.length-this.itemset_size + fixed.length;

                //pick all possible directions
                for(int i = first_position;i<=last_possible_path;i++){
                    int direction=transaction[i]% this.h;
                    Node next_node = ((InteriorNode) node).childs[direction];

                    //update prefix accordinly
                    int[] new_fixed= new int[fixed.length+1];
                    System.arraycopy(fixed,0,new_fixed, 0,fixed.length);
                    new_fixed[fixed.length]=transaction[i];

                    //if there is no next node then this path dies
                    if(next_node==null){
                        continue;
                    }
                    //if node is interior, advance the recursion with the new prefix
                    if(next_node instanceof InteriorNode){
                        this.update(next_node,transaction,i+1,new_fixed);
                    }
                    //else we are one item away from itemsets in the tree.
                    //guess what item we are missing and check against the itemsets
                    else {
                        for(int j =i+1; j<transaction.length;j++){
                            int[] new_new_fixed= new int[new_fixed.length+1];
                            System.arraycopy(new_fixed,0,new_new_fixed,0,fixed.length);
                            new_new_fixed[new_fixed.length]=transaction[j];

                            for(ItemSet itemset : ((LeafNode)next_node).itemsets){
                                if(Arrays.equals(itemset.itemset,new_new_fixed)){
                                    itemset.support++;
                                }
                            }
                        }
                    }
                }
            }
            public int get_candidate_count(){
                return this.population;
            }
        }

//============================================================================= MAPPER

        public void map(Object key, Text value, Context context
        ) throws IOException, InterruptedException {
            //I hate java
            Configuration conf = context.getConfiguration();

            float corr = conf.getFloat("corr",1);
            Integer min_supp= (int) Math.round(Float.parseFloat(conf.get("minsup"))*corr/10);

            String[] transaction_strings = value.toString().split("\n");
            List<int[]> transactions = new ArrayList<>(); //transactions in memory
            Map<Integer,Integer> occurences = new HashMap<>(); //dictionary for frequent items

            //We want our tree to have one branch at each interior node for every item in the alphabet
            //so we store the maximum item
            int branching_factor=0;

            for(String transaction_string : transaction_strings){
                String[] items = transaction_string.split(" ");
                int[] transaction = new int[items.length];
                for(int i=0; i< items.length;i++){
                    Integer item = Integer.parseInt(items[i]);

                    if(item>branching_factor) branching_factor=item;

                    transaction[i]=item;
                    Integer support = occurences.get(item);
                    if (support==null){
                        occurences.put(item,1);
                    }else {
                        occurences.put(item,support+1);
                    }
                }
                transactions.add(transaction);
            }
            //get frequent items
            ArrayList<Integer> frequent_items = new ArrayList<Integer>();
            for(Map.Entry<Integer,Integer> entry : occurences.entrySet()){
                if(entry.getValue()>=min_supp){
                    frequent_items.add(entry.getKey());
                    //emit (text) encodings of frequent items
                    word.set(entry.getKey().toString());
                    context.write(word, one);
                }
            }

            Collections.sort(frequent_items);

            if(frequent_items.size()==0) return;

            HashTree frequentsK = null; //stores frequent itemsets inbetween iterations. (here implicitly frequentsK = frequent_items)
            int k=1; // current itemset length (and tree height)

            while(true){ //main loop
                HashTree candidates = null;
                if(k==1) candidates = get_candidates_2(frequent_items); //generate frequent pairs separetely to avoid edge cases
                else     candidates = get_candidates_kp1(frequentsK);

                //calculate support for the new candidates in the tree
                if(candidates.get_candidate_count()==0) break;
                for (int[] transaction : transactions){
                    if(transaction.length >= k +1);
                    candidates.update_supports(transaction);
                }

                //filter the candidates in the tree that ended up being infrequent
                LeafNode node = candidates.last_leaf;
                while(node!= null){
                    ArrayList<ItemSet> filtered_itemsets = new ArrayList<>();
                    for(ItemSet itemset : node.itemsets){
                        if(itemset.support>=min_supp) {
                            word.set(itemset.itemset.toString().replaceAll("\\[|\\]|,", ""));
                            context.write(word,one);
                            filtered_itemsets.add(itemset);
                        }
                        else{
                            candidates.population-=1;
                        }
                    }
                    node.itemsets = filtered_itemsets;
                    node=node.next_leaf;
                }

                //we are done for the iteration. Use these frequent itemsets to generate the next tree of candidates
                k+=1;
                frequentsK=candidates;

                //if we have no frequents, then no candidates will be generated and we are done
                if(frequentsK.get_candidate_count()==0) break;
            }
        }

        //basic frequent pair candidate generation
        private HashTree get_candidates_2(ArrayList<Integer> frequent_items){
            HashTree candidate_tree = new HashTree(2, this.branching_factor);
            for (int i =0; i<frequent_items.size();i++)
                for (int j =i+1; j <frequent_items.size(); j++) {
                    int[] candidate = new int[]{frequent_items.get(i), frequent_items.get(j)};
                    candidate_tree.insert_candidate(new ItemSet(candidate));
                }
            return candidate_tree;
        }
        //Basically the same as candidate generation for normal apriori
        //Except by construction we get that all of the itemsets sharing k-1 items
        //live in the same leaf. So we don't need to combine all pairs of itemsets.
        private HashTree get_candidates_kp1(HashTree tree_k){
            int k = tree_k.itemset_size;
            HashTree candidates = new HashTree(k+1,branching_factor);
            LeafNode node=candidates.last_leaf;

            while(node!=null){
                for (int i=0;i<node.itemsets.size();i++){
                    ItemSet itemset1 = node.itemsets.get(i);
      nextcompare:  for(int j=i+1; j<node.itemsets.size();j++){
                        ItemSet itemset2 = node.itemsets.get(j);

                        for(int l=0;l<=k-1;l++){

                            // if the first k-1 do not match - disregard
                            if(l<k-1 && itemset1.itemset[l] != itemset2.itemset[l]) continue nextcompare;
                            if(l==k-1 && itemset1.itemset[l] < itemset2.itemset[l]){ //this means we got a candidate, join it!

                                int[] candidate = new int[k+1];
                                System.arraycopy(itemset1,0,candidate,0,k);
                                candidate[k]=itemset2.itemset[k-1];

                                //make sure all k subsets of our k+1 candidate are frequent
                                if (check_frequency_of_subsets(candidate,tree_k)) candidates.insert_candidate(new ItemSet(candidate));
                            }
                        }
                    }
                }
            }
            return candidates;
        }
        //same as normal apriori
        private boolean check_frequency_of_subsets(int[] candidate,HashTree tree_k_1){
            for(int i=0; i<candidate.length;i++){
                int[] first_part = Arrays.copyOfRange(candidate,0,i);
                int[] second_part = Arrays.copyOfRange(candidate,i+1,candidate.length);
                int[] subset = org.apache.commons.lang3.ArrayUtils.addAll(first_part,second_part);
                if (!tree_k_1.lookup(subset)) return false;
            }
            return true;
        }
    }

    public static class FilterPresentItemsetsReducer
            extends Reducer<Text,IntWritable,Text, NullWritable> {
        public void reduce(Text key, Iterable<IntWritable> values,
                           Context context
        ) throws IOException, InterruptedException {
            context.write(key,NullWritable.get());
        }
    }

    public static class SecondMapper extends Mapper<Object,Text,Text,IntWritable>{
        private List<int[]> candidates = new ArrayList<>(); //itemsets from cached file
        private int minsup;
        private Text word=new Text();
        private final IntWritable one = new IntWritable(1);

        public void setup(Context context) throws IOException{
            Configuration conf = context.getConfiguration();
            minsup = conf.getInt("minsup",Integer.MAX_VALUE);
            URI[] cached_files = context.getCacheFiles();
            BufferedReader readSet = new BufferedReader(new InputStreamReader(new FileInputStream(cached_files[0].toString())));

            //the itemsets are stored as "12 13 14" etc
            for (String itemset_str = readSet.readLine(); itemset_str != null; itemset_str=readSet.readLine()){
                String[] items_str = itemset_str.split(" ");
                int[] itemset = new int[items_str.length];
                for(int i=0;i<items_str.length;i++) itemset[i] = Integer.parseInt(items_str[i]);
                candidates.add(itemset);
            }

        }
        public void map(Object key, Text value, Context context) throws IOException,InterruptedException{
            //convert value to int array
            String[] items_str = value.toString().split(" ");
            HashSet<Integer> transcation = new HashSet<>();
            for(int i=0; i<items_str.length;i++) transcation.add(Integer.parseInt(items_str[i]));

            for(int[] candidate : candidates){
                if (is_subset(transcation,candidate)){
                    word.set(Arrays.toString(candidate));
                    context.write(word,one);
                }
            }
        }
    }
    public static class SecondReduce extends Reducer<Text,IntWritable,Text,IntWritable>{
        public void reduce(Text key, Iterable<IntWritable> values, Context context) throws InterruptedException, IOException{
            Configuration conf = context.getConfiguration();
            Integer min_supp= conf.getInt("minsup",Integer.MAX_VALUE);
            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }
            if(sum>=min_supp) context.write(key, new IntWritable(sum));
        }
    }

    private static boolean is_subset(HashSet<Integer> outer, int[] inner){
        int n = inner.length;
        for (int i = 0; i < n; i++) if (!outer.contains(inner[i])) return false;
        return true;
    }
}
