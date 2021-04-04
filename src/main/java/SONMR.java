import java.io.*;
import java.net.URI;
import java.util.*;
import java.util.stream.Collectors;

import com.nimbusds.jose.util.ArrayUtils;
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
import org.apache.hadoop.util.hash.Hash;
import org.apache.kerby.config.Conf;


import static org.apache.hadoop.mapreduce.lib.input.NLineInputFormat.setNumLinesPerSplit;

public class SONMR {
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
        conf.set("corr",corr_factor);
        job = Job.getInstance(conf, "second round");
        job.setJarByClass(SONMR.class);
        job.setMapperClass(SecondMapper.class);
        job.setMapOutputKeyClass(Text.class); //second mapper takes locally frequent itemsets, which we saved as Text
        job.setMapOutputValueClass(IntWritable.class);
        //job.setCombinerClass(FilterPresentItemsetsReducer.class); //not thinking of combiner for now
        job.setReducerClass(SecondReduce.class);
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
    public static class AprioriMapper
            extends Mapper<Object, Text, Text, IntWritable>{

        private Text word = new Text();
        private final static IntWritable one = new IntWritable(1);

        public void map(Object key, Text value, Context context
        ) throws IOException, InterruptedException {
            Configuration conf = context.getConfiguration();
            float corr = conf.getFloat("corr",1);
            Integer min_supp= (int) Math.round(Float.parseFloat(conf.get("minsup"))*corr/10);

            String[] transaction_strings = value.toString().split("\n");
            List<HashSet<Integer>> transactions = new ArrayList<>();
            ArrayList<int[]> frequent_itemsets = new ArrayList<int[]>();
            Map<Integer,Integer> occurences = new HashMap<>();

            for(String transaction_string : transaction_strings){
                String[] items = transaction_string.split(" ");
                HashSet<Integer> transaction = new HashSet<Integer>();
                for(int i=0; i< items.length;i++){
                    Integer item = Integer.parseInt(items[i]);
                    transaction.add(item);
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
            List<int[]> frequentsK = null;
            int k=1; // current itemset length

            while(true){
                List<int[]> candidates; //will store candidates of length k+1
                if(k==1) candidates = get_candidates_2(frequent_items);
                else     candidates = get_candidates_kp1(frequentsK,k);

                k+=1;
                //filter the candidate to obtain new frequent itemsets for the next round of candidate generation
                frequentsK = new ArrayList<int[]>();
                for (int[] candidate : candidates){
                    if (check_candidate_support(candidate,transactions,min_supp)) {
                        frequentsK.add(candidate);
                        //emit the result (string-encoded)
                        //https://stackoverflow.com/a/34070570
                        word.set(Arrays.toString(candidate).replaceAll("\\[|\\]|,", ""));
                        context.write(word,one);
                        frequent_itemsets.add(candidate);
                    }
                }
                //if no new frequent items were obtained we are done
                if (frequentsK.size()==0) break;
            }
        }
        //joins all frequent items to get ordered frequent pairs
        private List<int[]> get_candidates_2(ArrayList<Integer> frequent_items){
            List<int[]> candidates = new ArrayList<int[]>();
            for(int i=0;i<frequent_items.size();i++){
                Integer item1 = frequent_items.get(i);
                for (int j=i+1;j<frequent_items.size();j++){
                    Integer item2 = frequent_items.get(j);
                    int[] candidate = new int[] {item1,item2};
                    candidates.add(candidate);
                }
            }
            return candidates;
        }

        //join frequent itemsets of length k sharing the first k-1 items to obtain candidates k+1
        private List<int[]> get_candidates_kp1(List<int[]> frequentsK,int k){
            List<int[]> candidates = new ArrayList<int[]>();

            for(int i =0;i<frequentsK.size(); i++){
                int[] itemset1 = frequentsK.get(i);
   nextcompare: for(int j =i+1; j<frequentsK.size();j++){
                    int[] itemset2 = frequentsK.get(j);
                    //check that
                    for(int l=0;l<=k-1;l++){

                        // if the first k-1 do not match - disregard
                        if(l<k-1 && itemset1[l] != itemset2[l]) continue nextcompare;
                        if(l==k-1 && itemset1[l] < itemset2[l]){ //this means we got a candidate, join it!
                            int[] candidate = new int[k+1];
                            System.arraycopy(itemset1,0,candidate,0,k);
                            candidate[k]=itemset2[k-1];

                            //make sure all k subsets of our k+1 candidate are frequent
                            if (check_frequency_of_subsets(candidate,frequentsK)) candidates.add(candidate);
                        }

                    }
                }
            }
            return candidates;
        }

        private boolean check_frequency_of_subsets(int[] candidate, List<int[]> frequentsK_1){
            //converting solely for list.contains(). lazy
            ArrayList<int[]> frequentsk_1 = new ArrayList<int[]>();
            for(int[] itemset : frequentsK_1)
                frequentsk_1.add(itemset);
            //remove ith item to obtain an immediate subset
            for(int i=0; i<candidate.length;i++){
                int[] first_part = Arrays.copyOfRange(candidate,0,i);
                int[] second_part = Arrays.copyOfRange(candidate,i+1,candidate.length);
                int[] subset = org.apache.commons.lang3.ArrayUtils.addAll(first_part,second_part);
                if(!frequentsk_1.contains(subset)) return false;
            }
            return true;
        }

        private boolean check_candidate_support(int[] item_set,List<HashSet<Integer>> transactions,int minsup){
            int support=0;
            for (HashSet<Integer> transaction : transactions){
                if (transaction.size() < item_set.length) continue;
                if (is_subset(transaction, item_set)) support++;
            }
            return (support>=minsup);
        }
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

