package project.core.original.javacode; /**
 * Created by root on 2015/12/1.
 */

import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RecordWriter;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.util.Progressable;

import java.io.DataOutputStream;
import java.io.IOException;

public class BinaryOutputFormat extends FileOutputFormat<BytesWritable, BytesWritable> {
    @Override
    public RecordWriter<BytesWritable, BytesWritable> getRecordWriter
            (FileSystem ignored, JobConf job, String name, Progressable progress) throws IOException
    {
        Path file = FileOutputFormat.getTaskOutputPath(job, name);
        FileSystem fs = file.getFileSystem(job);
        FSDataOutputStream fileOut = fs.create(file, progress);
        return new ByteRecordWriter(fileOut);
    }

    protected static class ByteRecordWriter implements RecordWriter<BytesWritable,BytesWritable >
    {
        private DataOutputStream out;
        private boolean nullValue, nullKey;

        public ByteRecordWriter(DataOutputStream out)
        {
            this.out = out;
        }
        public synchronized  void write(BytesWritable key, BytesWritable value) throws IOException
        {
            nullKey = key.getLength() == 0;
            nullValue = value.getLength() == 0;

            if (!nullKey){
                out.write( key.getBytes(), 0, key.getLength() );
            }
            if (!nullValue){
                out.write( value.getBytes(), 0, value.getLength() );
            }
        }

        public synchronized void close(Reporter reporter) throws IOException {
            out.close();
        }
    }
}