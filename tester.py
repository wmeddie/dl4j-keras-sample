
class JvmMetaClass(type):
    def __call__(cls, *args, **kwargs):
        try:
            from pyspark.context import SparkContext
            sc = SparkContext._active_spark_context
            if sc is not None:    
                names = cls.jvm_cls_name.split('.')
                last = sc._jvm
                for name in names:
                    last = getattr(last, name)
                return last(*args, **kwargs)
            else:
                from jnius import autoclass
                return autoclass(cls.jvm_cls_name)(*args, **kwargs)
        except:
            from jnius import autoclass
            return autoclass(cls.jvm_cls_name)(*args, **kwargs)

class File(object):
    __metaclass__ = JvmMetaClass
    jvm_cls_name = 'java.io.File'


if __name__ == '__main__':
        from pyspark.context import SparkContext
        sc = SparkContext()
	f = File("tester.py")
        print(f)
	print(f.toString())

