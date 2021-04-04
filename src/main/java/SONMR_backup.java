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

import java.io.*;
import java.net.URI;
import java.util.*;

import static org.apache.hadoop.mapreduce.lib.input.NLineInputFormat.setNumLinesPerSplit;

public class SONMR_backup {
    public static void main(String[] args) throws Exception {
        int dataset_size= Integer.parseInt(args[0]);
        int transactions_per_block= Integer.parseInt(args[1]);
        String min_supp= args[2];
        float corr_factor=Float.parseFloat(args[3]);

        Configuration conf = new Configuration();
        conf.set("minsup", min_supp);
        Job job = Job.getInstance(conf, "first round");
        job.setJarByClass(SONMR_backup.class);
        job.setMapperClass(AprioriMapper.class);
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(IntWritable.class);
        //job.setCombinerClass(FilterPresentItemsetsReducer.class);
        job.setReducerClass(FilterPresentItemsetsReducer.class);
        job.setInputFormatClass(MultiLineInputFormat.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(NullWritable.class);
        setNumLinesPerSplit(job, transactions_per_block);
        FileInputFormat.addInputPath(job, new Path(args[4]));
        FileOutputFormat.setOutputPath(job, new Path(args[5]));

        job.waitForCompletion(false);

        conf = new Configuration();
        conf.set("minsup", min_supp);
        job = Job.getInstance(conf, "second round");
        job.setJarByClass(SONMR_backup.class);
        job.setMapperClass(SecondMapper.class);
        job.setMapOutputKeyClass(Text.class); //second mapper takes locally frequent itemsets, which we saved as Text
        job.setMapOutputValueClass(IntWritable.class);
        //job.setCombinerClass(FilterPresentItemsetsReducer.class); //not thinking of combiner for now
        job.setReducerClass(FilterPresentItemsetsReducer.class);
        //job.setInputFormatClass(MultiLineInputFormat.class); we are using default one-line per mapper input
        job.setOutputKeyClass(Text.class); //the key is an itemset, i.e Text
        job.setOutputValueClass(IntWritable.class); //the value is the support, IntWritable
        FileInputFormat.addInputPath(job, new Path(args[4])); //the input is dataset
        FileOutputFormat.setOutputPath(job, new Path(args[6])); //the output path is the output directory
        Path candidates_path = new Path(args[5]+ File.separator + "part-r-00000");
        job.addCacheFile(candidates_path.toUri());
        System.exit(job.waitForCompletion(true) ? 0:1);
    }
    public static class AprioriMapper
            extends Mapper<Object, Text, Text, IntWritable>{

        private Text word = new Text();
        private final static IntWritable one = new IntWritable(1);

        //wrapper for itemsets to keep support information local
        public class ItemSet{
            int[] itemset;
            int support=0;
            ItemSet(int[] itemset){
                this.itemset=itemset;
            }
        }

        public void map(Object key, Text value, Context context
        ) throws IOException, InterruptedException {
            //I hate java
            Configuration conf = context.getConfiguration();
            Integer min_supp= (int) Math.round(Float.parseFloat(conf.get("minsup"))/10);

            String[] transaction_strings = value.toString().split("\n");
            ArrayList<int[]> transactions = new ArrayList<>();
            ArrayList<int[]> frequent_itemsets = new ArrayList<int[]>();
            Map<Integer,Integer> occurences = new HashMap<>();

            for(String transaction_string : transaction_strings){
                String[] items = transaction_string.split(" ");
                int[] transaction = new int[items.length];
                for(int i=0; i< items.length;i++){
                    Integer item = Integer.parseInt(items[i]);
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
                    //emit string(text) encodings of frequent items
                    word.set(entry.getKey().toString());
                    context.write(word, one);
                }
            }
            Collections.sort(frequent_items);
            if(frequent_items.size()==0) return;

            //stores frequent itemsets inbetween iterations. (here implicitly frequentsK = frequent_items)
            List<ItemSet> frequentsK=null;
            int k=1; // current itemset length

            while(true){
                List<ItemSet> candidates; //will store candidates of length k+1
                if(k==1) candidates = get_candidates_2(frequent_items);
                else     candidates = get_candidates_kp1(frequentsK,k);

                k+=1;
                //filter the candidate to obtain new frequent itemsets for the next round of candidate generation
                frequentsK= new ArrayList<ItemSet>();
                for (ItemSet candidate : candidates){
                    if (check_candidate_support(candidate,transactions,min_supp)) {
                        frequentsK.add(candidate);
                        //emit the result (string-encoded)
                        //https://stackoverflow.com/a/34070570
                        word.set(Arrays.toString(candidate.itemset).replaceAll("\\[|\\]|,", ""));
                        context.write(word,one);
                        frequent_itemsets.add(candidate.itemset);
                    }
                }
                //if no new frequent items were obtained we are done
                if (frequentsK.size()==0) break;
            }
        }
        //joins all frequent items to get ordered frequent pairs
        private List<ItemSet> get_candidates_2(ArrayList<Integer> frequent_items){
            List<ItemSet> candidates = new ArrayList<ItemSet>();
            for(int i=0;i<frequent_items.size();i++){
                Integer item1 = frequent_items.get(i);
                for (int j=i+1;j<frequent_items.size();j++){
                    Integer item2 = frequent_items.get(j);
                    int[] candidate = new int[] {item1,item2};
                    ItemSet icandidate = new ItemSet(candidate);
                    candidates.add(icandidate);
                }
            }
            return candidates;
        }

        //join frequent itemsets of length k sharing the first k-1 items to obtain candidates k+1
        private List<ItemSet> get_candidates_kp1(List<ItemSet> frequentsK,int k){
            List<ItemSet> candidates = new ArrayList<ItemSet>();

            for(int i =0;i<frequentsK.size(); i++){
                ItemSet itemset1 = frequentsK.get(i);
   nextcompare: for(int j =i+1; j<frequentsK.size();j++){
                    ItemSet itemset2 = frequentsK.get(j);
                    //check that
                    for(int l=0;l<=k-1;l++){

                        // if the first k-1 do not match - disregard
                        if(l<k-1 && itemset1.itemset[l] != itemset2.itemset[l]) continue nextcompare;
                        if(l==k-1 && itemset1.itemset[l] < itemset2.itemset[l]){ //this means we got a candidate, join it!
                            int[] candidatear = new int[k+1];
                            System.arraycopy(itemset1.itemset,0,candidatear,0,k);
                            candidatear[k]=itemset2.itemset[k-1];

                            //make sure all k subsets of our k+1 candidate are frequent
                            if (check_frequency_of_subsets(candidatear,frequentsK)) candidates.add(new ItemSet(candidatear));
                        }

                    }
                }
            }
            return candidates;
        }

        private boolean check_frequency_of_subsets(int[] candidate, List<ItemSet> frequentsK_1){
            //unwrap ItemSets into int arrays. I know this is a mess, but it is worth it, especially for hashtree implementation
            ArrayList<int[]> frequentsk_1 = new ArrayList<int[]>();
            for(ItemSet itemset : frequentsK_1)
                frequentsk_1.add(itemset.itemset);
            //remove ith item to obtain an immediate subset
            for(int i=0; i<candidate.length;i++){
                int[] first_part = Arrays.copyOfRange(candidate,0,i);
                int[] second_part = Arrays.copyOfRange(candidate,i+1,candidate.length);
                int[] subset = org.apache.commons.lang3.ArrayUtils.addAll(first_part,second_part);
                if(!frequentsk_1.contains(subset)) return false;
            }
            return true;
        }

        private boolean check_candidate_support(ItemSet item_set,ArrayList<int[]> transactions,int minsup){
            int support=0;
            for (int[] transaction : transactions){
                if (transaction.length < item_set.itemset.length) continue;
                if (is_subset(transaction, item_set.itemset)) support++;
            }
            return (support>=minsup);
        }

        //https://www.geeksforgeeks.org/find-whether-an-array-is-subset-of-another-array-set-1/
    }



    public static class FilterPresentItemsetsReducer
            extends Reducer<Text,IntWritable,Text,NullWritable> {
        public void reduce(Text key, Iterable<IntWritable> values,
                           Context context
        ) throws IOException, InterruptedException {
            context.write(key,NullWritable.get());
        }
    }

    public static class SecondMapper extends Mapper<Object,Text,Text,IntWritable>{
        private List<int[]> candidates = new ArrayList<>();
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
            try{
                setup(context);
            } catch(IOException e){
                throw new IOException();
            }
            //convert value to int array
            String[] items_str = value.toString().split(" ");
            int[] transcation = new int[items_str.length];
            for(int i=0; i<transcation.length;i++) transcation[i]=Integer.parseInt(items_str[i]);

            for(int[] candidate : candidates){
                if (is_subset(transcation,candidate)){
                    word.set(Arrays.toString(candidate));
                    context.write(word,one);
                }
            }
        }
    }
    public static class SecondReduce extends Reducer<Text,IntWritable,Text,IntWritable>{
        public void map(Text key, Iterable<IntWritable> values, Context context) throws InterruptedException, IOException{
            Configuration conf = context.getConfiguration();
            Integer min_supp= conf.getInt("minsup",Integer.MAX_VALUE);
            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }
            if(sum>=min_supp) context.write(key, new IntWritable(sum));
        }
    }

    private static boolean is_subset(int[] outer, int[] inner){
        int m = outer.length;
        int n = inner.length;
        HashSet<Integer> hset = new HashSet<>();

        // hset stores all the values of arr1
        for (int i = 0; i < m; i++) {
            if (!hset.contains(outer[i]))
                hset.add(outer[i]);
        }

        // loop to check if all elements
        //  of arr2 also lies in arr1
        for (int i = 0; i < n; i++)
        {
            if (!hset.contains(inner[i]))
                return false;
        }
        return true;
    }
}

